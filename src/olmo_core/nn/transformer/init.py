import math
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

if TYPE_CHECKING:
    from ..attention import SequenceMixer
    from ..feed_forward import FeedForward
    from ..moe import MoEBase


def _apply_init(init_fun, x: torch.Tensor, *args, **kwargs):
    if not isinstance(x, DTensor):
        init_fun(x, *args, **kwargs)
        return

    # Initialize full version of x locally, then apply init to that.
    full_x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    init_fun(full_x, *args, **kwargs)
    full_x = distribute_like(x, full_x)

    # Now copy over the corresponding shard of `full_x` into `x`.
    get_local_tensor(x).copy_(get_local_tensor(full_x))


def init_linear(
    m: nn.Linear | nn.Conv1d, *, std: float = 0.02, generator: Optional[torch.Generator] = None
):
    _apply_init(
        nn.init.trunc_normal_,
        m.weight,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class InitMethod(StrEnum):
    normal = "normal"
    """
    Every linear and embedding layer and initialized from a truncated normal distributed
    with standard deviation 0.02.
    """

    normalized = "normalized"
    """
    Follow the nGPT initialization scheme.
    """

    llama = "llama"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the number of layers.
    """

    llama_depth = "llama_depth"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the layer index.
    """

    fan_in = "fan_in"
    """
    Per-layer fan-in initialization where each weight matrix is initialized with
    ``std = 1/√d_in`` where ``d_in`` is the fan-in (number of input features) of that
    specific layer. Embeddings use ``std = 1.0`` with normal distribution.
    This provides forward-pass variance-preserving initialization adapted to each layer's
    specific dimensions, with no depth scaling.
    """

    complete_p = "complete_p"
    """
    CompleteP (Complete Parametrization) initialization designed for Adam-type optimizers.

    Each dense hidden weight (excluding the input embedding) is initialized with standard
    deviation ``init_std * sqrt(d_in / d_base)`` where ``d_in`` is the fan-in of the weight
    in the target model and ``d_base`` is the corresponding fan-in in the base reference model.
    The readout (LM-head) weight uses a constant ``init_std`` with no width scaling.

    These init stds are paired with explicit per-weight output multipliers ``d_base / d_in``
    (implemented in :class:`~olmo_core.nn.feed_forward.CompletePFeedForward` and
    :class:`~olmo_core.nn.transformer.block.CompletePReorderedNormTransformerBlock`) so that
    the effective output variance is width-independent across model sizes.

    Residual branches are additionally scaled by ``L_base / L`` (set via
    ``attention_residual_alpha`` and ``feed_forward_residual_alpha`` in the block config).

    Requires ``d_model_base`` to be stored on :class:`~olmo_core.nn.transformer.model.Transformer`
    and passed through ``init_attention``; feed-forward base dimensions are read directly from
    :class:`~olmo_core.nn.feed_forward.CompletePFeedForward` attributes.
    """

    complete_p_sde = "complete_p_sde"
    """
    Like :data:`complete_p`, but every hidden weight -- including the residual-writing
    "down" projections ``w2`` (feed-forward) and ``w_out`` (attention) -- is initialized with
    ``std = init_std * sqrt(d_model / d_model_base)``, i.e. the *residual* width sets the init
    scale for all of them rather than each weight's own fan-in.

    Multipliers are unchanged from :data:`complete_p`: ``d_base / d_in`` per weight, times
    ``L_base / L`` on the residual branches. So the down projections have an effective map of
    ``(1 / (L * M)) W`` with ``W_ij ~ O(sqrt(N))``, where ``N = d_model``, ``L = n_layers``, and
    ``M`` is the block hidden width (``hidden_size`` for the MLP, ``n_heads`` for attention).

    Because the coherent (Adam-driven) response of a layer grows like its fan-in while the
    incoherent (init) response grows like the square root of its fan-in, this opens a gap between
    the two: at init each block writes ``Theta(sqrt(alpha / L))`` into the residual stream while
    each Adam step moves it by ``Theta(eta / L)``. Summed over ``L`` blocks -- diffusively for the
    former, coherently for the latter -- both are ``Theta(1)`` with a learning rate that is
    constant in ``N``, ``M`` and ``L``. The residual stream then has a joint SDE limit with
    diffusion coefficients

    - ``alpha_mlp = d_model / (hidden_size * n_layers)``
    - ``alpha_att = d_model / (n_heads * head_dim * n_layers)``

    .. important::
        This only has any effect on a block whose residual branch output is *not* normalized,
        i.e. :data:`~olmo_core.nn.transformer.block.TransformerBlockType.default` (pre-norm).
        RMSNorm is 0-homogeneous, so the branch-output norm in ``reordered_norm`` / ``peri_norm``
        cancels the init std of the down projections exactly and this init method degenerates
        back to :data:`complete_p`.
    """

    def init_embeddings(
        self,
        m: nn.Embedding,
        *,
        d_model: int,
        embed_scale: Optional[float] = None,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth):
            _apply_init(nn.init.normal_, m.weight, generator=generator)
        elif self == InitMethod.normalized:
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=d_model**-0.5)
        elif self == InitMethod.fan_in:
            # Fan-in init uses std = 1.0 for embeddings, scaled down by embed_scale if set
            emb_std = 1.0 / embed_scale if embed_scale is not None else 1.0
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=emb_std)
        else:
            # Covers InitMethod.normal, InitMethod.complete_p and InitMethod.complete_p_sde.
            # CompleteP does not scale embedding init; embeddings are excluded from the
            # width-scaling rule.
            _apply_init(
                nn.init.trunc_normal_,
                m.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )

    def init_final_w_out(
        self,
        m: nn.Linear,
        *,
        d_model: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (
            InitMethod.llama,
            InitMethod.llama_depth,
            InitMethod.normalized,
            InitMethod.fan_in,
        ):
            std = d_model**-0.5
        # For complete_p: readout uses constant std (no d_in scaling).
        # The explicit d_model_base/d_model multiplier is applied in the forward pass.
        init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: "SequenceMixer",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        m.init_weights(
            init_method=self,
            d_model=d_model,
            block_idx=block_idx,
            num_blocks=num_blocks,
            std=std,
            generator=generator,
        )

    def init_feed_forward(
        self,
        m: "FeedForward",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        from ..feed_forward import CompletePFeedForward

        if self in (InitMethod.complete_p, InitMethod.complete_p_sde) and isinstance(
            m, CompletePFeedForward
        ):
            # CompleteP: each weight is initialized with std = init_std * sqrt(d_in / d_base),
            # paired with the explicit per-weight output multipliers in CompletePFeedForward.forward.
            w1_std = std * math.sqrt(d_model / m.d_model_base)
            if self == InitMethod.complete_p_sde:
                # SDE variant: the down projection is initialized off the *residual* width rather
                # than its own fan-in, so all three weights share the same std. Paired with the
                # unchanged w2 multiplier (hidden_size_base / hidden_size, times L_base / L on the
                # residual branch) this makes the block's init contribution to the residual stream
                # diffusive, O(1/sqrt(L)), while its Adam-driven contribution stays O(1/L).
                w2_std = w1_std
            else:
                w2_std = std * math.sqrt(m.hidden_size / m.hidden_size_base)
            # w3 has the same fan-in as w1 (d_model → hidden_size)
            init_linear(m.w1, std=w1_std, generator=generator)
            init_linear(m.w3, std=w1_std, generator=generator)
            init_linear(m.w2, std=w2_std, generator=generator)
            return

        # Compute std for w1 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w1 uses 1/√d_in where d_in = d_model (ignores base std parameter)
            std = m.w1.in_features**-0.5
        elif self == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(m.w1, std=std, generator=generator)

        # Compute std for w3 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w3 uses 1/√d_in where d_in = d_model
            std = m.w3.in_features**-0.5
        elif self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        init_linear(m.w3, std=std, generator=generator)

        # Compute std for w2 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w2 uses 1/√d_in where d_in = hidden_size
            std = m.w2.in_features**-0.5
        elif self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(m.w2, std=std, generator=generator)

    def init_feed_forward_moe(
        self,
        m: "MoEBase",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        from ..moe import DroplessMoEMLP, MoELinearRouter, MoEMLP

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif self == InitMethod.fan_in:
            # For fan_in, router weight uses 1/√d_model
            std = d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            cast(MoELinearRouter, m.router).weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        mlp = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp)

        # Initialize w1 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w2 (maps hidden_size -> d_model, fan-in = hidden_size)
        if self == InitMethod.fan_in:
            std = mlp.hidden_size**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w3 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )
