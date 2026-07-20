"""
Joint width/depth/hidden-width scaling rules for the SDE parametrization.

The parametrization implemented here (see
:data:`~olmo_core.nn.transformer.init.InitMethod.complete_p_sde`) writes every residual branch as

.. code::

    h_up   = (1 / N) W_up h,                 W_up[i,j]   ~ O(sqrt(N))
    h_next = h + (1 / (L * M)) W_down phi(h_up),  W_down[i,j] ~ O(sqrt(N))

where ``N = d_model``, ``L = n_layers``, and ``M`` is the hidden width of the block --
``hidden_size`` for the MLP and ``n_heads * head_dim`` for attention (i.e. the fan-in of
``w_out``; at fixed ``head_dim`` it is the head *count* that scales).

Because Adam's update is entrywise scale-invariant, the init std and the learning rate are
independent knobs, and the ``sqrt(N)`` init std on the *down* projection buys a gap between the
incoherent (init) and coherent (Adam) responses of a block:

- at init, each block writes ``Theta(sqrt(alpha / L))`` into the residual stream, and the
  ``L`` contributions add diffusively to ``Theta(sqrt(alpha))``;
- each Adam step moves each block's contribution by ``Theta(eta / L)``, and the ``L``
  contributions add coherently to ``Theta(eta)``.

Both are ``Theta(1)`` with a learning rate that is constant in ``N``, ``M`` and ``L``. The
residual stream therefore has a joint SDE limit whose diffusion coefficients are exactly

.. code::

    alpha_mlp = N / (M * L)          = d_model / (hidden_size * n_layers)
    alpha_att = N / (H * d_head * L) = d_model / (n_heads * head_dim * n_layers)

Holding both alphas fixed while growing ``N``, ``M``, ``H`` and ``L`` gives the SDE ray
(:meth:`JointScalingConfig.scale_sde`). Any other joint scaling is also stable -- the drift is
``Theta(eta)`` regardless -- it just changes the diffusion coefficients. In particular a linear
joint scaling ``N ~ M ~ H ~ L ~ s`` (:meth:`JointScalingConfig.scale_linear`) sends both alphas to
zero like ``1/s``, smoothly degenerating the model toward the drift-only (ODE) limit.

Example
-------

.. code:: python

    base = JointScalingConfig.base(d_model=768, n_layers=12, n_heads=12, head_dim=64)

    # Hold both alphas fixed: N x4, L x2  =>  M x2, H x2.
    scaling = base.scale_sde(width_mult=4, depth_mult=2)

    # Or scale everything linearly, sending both alphas to alpha_base / 2.
    scaling = base.scale_linear(mult=2)

    model_config = TransformerConfig.llama_like(
        d_model=scaling.d_model,
        n_layers=scaling.n_layers,
        n_heads=scaling.n_heads,
        head_dim=scaling.head_dim,
        feed_forward=FeedForwardConfig(hidden_size=scaling.hidden_size, bias=False),
        ...,
    )
    scaling.apply(model_config)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import ensure_multiple_of

from ..attention import AttentionConfig
from ..feed_forward import FeedForwardConfig, FeedForwardType
from .config import TransformerBlockType, TransformerConfig
from .init import InitMethod

__all__ = ["JointScalingConfig"]

log = logging.getLogger(__name__)


@dataclass
class JointScalingConfig(Config):
    """
    A (base model, target model) pair for the joint SDE parametrization, plus the machinery to
    stamp the resulting multipliers and init rules onto a :class:`TransformerConfig`.

    All multipliers are ratios to the base model, so a target model that *is* the base model is
    a fixed point: every multiplier is ``1.0`` and training is identical to standard training.
    """

    d_model_base: int
    """Residual width ``N`` of the base model."""

    n_layers_base: int
    """Depth ``L`` of the base model."""

    n_heads_base: int
    """Head count ``H`` of the base model."""

    hidden_size_base: int
    """MLP hidden width ``M`` of the base model."""

    head_dim: int
    """
    Attention head dimension. Held **fixed** across the sweep -- it is the head *count* that
    scales, so that the fan-in of ``w_out`` is proportional to ``n_heads``.
    """

    d_model: int
    """Residual width ``N`` of the target model."""

    n_layers: int
    """Depth ``L`` of the target model."""

    n_heads: int
    """Head count ``H`` of the target model."""

    hidden_size: int
    """MLP hidden width ``M`` of the target model."""

    @classmethod
    def base(
        cls,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        hidden_size: Optional[int] = None,
        hidden_size_multiple_of: int = 256,
    ) -> "JointScalingConfig":
        """
        Build the identity scaling, i.e. the base model as its own target.

        :param head_dim: Defaults to ``d_model // n_heads``, i.e. the base model is a conventional
            transformer. It is then held fixed by :meth:`scale_sde` / :meth:`scale_linear`.
        :param hidden_size: Defaults to the llama-like ``8/3 * d_model`` rounded up to a multiple
            of ``hidden_size_multiple_of``, matching :meth:`TransformerConfig.llama_like`.
        """
        if head_dim is None:
            head_dim = d_model // n_heads
        if hidden_size is None:
            hidden_size = ensure_multiple_of(int(8 * d_model / 3), hidden_size_multiple_of)
        return cls(
            d_model_base=d_model,
            n_layers_base=n_layers,
            n_heads_base=n_heads,
            hidden_size_base=hidden_size,
            head_dim=head_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
        )

    def _retarget(
        self,
        *,
        width_mult: float,
        depth_mult: float,
        hidden_mult: float,
        hidden_size_multiple_of: int,
        n_heads_multiple_of: int,
    ) -> "JointScalingConfig":
        return JointScalingConfig(
            d_model_base=self.d_model_base,
            n_layers_base=self.n_layers_base,
            n_heads_base=self.n_heads_base,
            hidden_size_base=self.hidden_size_base,
            head_dim=self.head_dim,
            d_model=ensure_multiple_of(round(self.d_model_base * width_mult), self.head_dim),
            n_layers=round(self.n_layers_base * depth_mult),
            n_heads=ensure_multiple_of(
                round(self.n_heads_base * hidden_mult), n_heads_multiple_of
            ),
            hidden_size=ensure_multiple_of(
                round(self.hidden_size_base * hidden_mult), hidden_size_multiple_of
            ),
        )

    def scale_sde(
        self,
        *,
        width_mult: float,
        depth_mult: float,
        hidden_size_multiple_of: int = 256,
        n_heads_multiple_of: int = 1,
    ) -> "JointScalingConfig":
        """
        Scale along the SDE ray: hold :attr:`alpha_mlp` and :attr:`alpha_att` fixed.

        Growing ``N`` by ``width_mult`` and ``L`` by ``depth_mult`` forces ``M`` and ``H`` to grow
        by ``width_mult / depth_mult``. Rounding ``M`` and ``H`` to usable sizes perturbs the alphas
        slightly; the *achieved* alphas are what :meth:`apply` uses and what :meth:`summary`
        reports, so check them.

        :param width_mult: Multiplier on ``d_model`` relative to the base model.
        :param depth_mult: Multiplier on ``n_layers`` relative to the base model.
        """
        if width_mult <= 0 or depth_mult <= 0:
            raise OLMoConfigurationError("'width_mult' and 'depth_mult' must be positive")
        return self._retarget(
            width_mult=width_mult,
            depth_mult=depth_mult,
            hidden_mult=width_mult / depth_mult,
            hidden_size_multiple_of=hidden_size_multiple_of,
            n_heads_multiple_of=n_heads_multiple_of,
        )

    def scale_linear(
        self,
        *,
        mult: float,
        hidden_size_multiple_of: int = 256,
        n_heads_multiple_of: int = 1,
    ) -> "JointScalingConfig":
        """
        Scale ``N``, ``M``, ``H`` and ``L`` all linearly by ``mult``, in the same parametrization.

        This leaves the ``Theta(eta)`` drift intact but drives both alphas down like ``1 / mult``,
        so the model's initialization becomes progressively less diffusive and the depth limit
        degenerates toward a drift-only ODE. Useful as the contrast case for :meth:`scale_sde`.
        """
        if mult <= 0:
            raise OLMoConfigurationError("'mult' must be positive")
        return self._retarget(
            width_mult=mult,
            depth_mult=mult,
            hidden_mult=mult,
            hidden_size_multiple_of=hidden_size_multiple_of,
            n_heads_multiple_of=n_heads_multiple_of,
        )

    @property
    def width_mult(self) -> float:
        """The *achieved* width multiplier ``N / N_base``, after rounding."""
        return self.d_model / self.d_model_base

    @property
    def depth_mult(self) -> float:
        """The *achieved* depth multiplier ``L / L_base``, after rounding."""
        return self.n_layers / self.n_layers_base

    @property
    def hidden_mult(self) -> float:
        """The *achieved* MLP hidden-width multiplier ``M / M_base``, after rounding."""
        return self.hidden_size / self.hidden_size_base

    @property
    def head_mult(self) -> float:
        """The *achieved* head-count multiplier ``H / H_base``, after rounding."""
        return self.n_heads / self.n_heads_base

    @property
    def alpha_mlp(self) -> float:
        """The MLP diffusion coefficient ``d_model / (hidden_size * n_layers)``."""
        return self.d_model / (self.hidden_size * self.n_layers)

    @property
    def alpha_att(self) -> float:
        """The attention diffusion coefficient ``d_model / (n_heads * head_dim * n_layers)``."""
        return self.d_model / (self.n_heads * self.head_dim * self.n_layers)

    @property
    def alpha_mlp_base(self) -> float:
        """:attr:`alpha_mlp` of the base model."""
        return self.d_model_base / (self.hidden_size_base * self.n_layers_base)

    @property
    def alpha_att_base(self) -> float:
        """:attr:`alpha_att` of the base model."""
        return self.d_model_base / (self.n_heads_base * self.head_dim * self.n_layers_base)

    @property
    def residual_alpha(self) -> float:
        """The residual branch multiplier ``L_base / L``."""
        return self.n_layers_base / self.n_layers

    @property
    def is_base(self) -> bool:
        """Whether the target model *is* the base model, in which case every multiplier is 1."""
        return (
            self.d_model == self.d_model_base
            and self.n_layers == self.n_layers_base
            and self.n_heads == self.n_heads_base
            and self.hidden_size == self.hidden_size_base
        )

    def summary(self) -> str:
        """A one-line summary of the target model and its achieved alphas, for logging."""
        return (
            f"N={self.d_model} (x{self.d_model / self.d_model_base:.3g}) "
            f"L={self.n_layers} (x{self.n_layers / self.n_layers_base:.3g}) "
            f"M={self.hidden_size} (x{self.hidden_size / self.hidden_size_base:.3g}) "
            f"H={self.n_heads} (x{self.n_heads / self.n_heads_base:.3g}) "
            f"d_head={self.head_dim} | "
            f"alpha_mlp={self.alpha_mlp:.4g} (base {self.alpha_mlp_base:.4g}) "
            f"alpha_att={self.alpha_att:.4g} (base {self.alpha_att_base:.4g})"
        )

    def apply(self, config: TransformerConfig) -> TransformerConfig:
        """
        Stamp the parametrization onto ``config``, in place. This sets the ``complete_p_sde`` init
        method, swaps in a :class:`~olmo_core.nn.feed_forward.CompletePFeedForward`, fills in the
        base dimensions on the attention / feed-forward / LM-head configs, and sets the residual
        branch multipliers to ``L_base / L``.

        ``config`` must already have been built at the target ``d_model`` / ``n_layers`` /
        ``n_heads`` / ``head_dim`` / ``hidden_size``.

        :returns: ``config``, for chaining.

        :raises OLMoConfigurationError: If ``config``'s dimensions disagree with this scaling, or if
            its block type normalizes the residual branch output (which would cancel the
            parametrization -- see :data:`InitMethod.complete_p_sde`).
        """
        if config.d_model != self.d_model or config.n_layers != self.n_layers:
            raise OLMoConfigurationError(
                f"model config (d_model={config.d_model}, n_layers={config.n_layers}) does not "
                f"match this scaling (d_model={self.d_model}, n_layers={self.n_layers})"
            )

        # The whole point of the sqrt(N) init on the down projections is the gap it opens between
        # the init and Adam responses of a block. RMSNorm is 0-homogeneous, so a norm on the branch
        # *output* divides that gap right back out and the parametrization silently degenerates.
        if config.block.name != TransformerBlockType.default:
            raise OLMoConfigurationError(
                f"the SDE parametrization requires block_name='default' (pre-norm), got "
                f"'{config.block.name}'. Blocks that normalize the residual branch output "
                f"(reordered_norm, peri_norm) cancel the init std of the down projections exactly."
            )

        attention = config.block.sequence_mixer
        if not isinstance(attention, AttentionConfig):
            raise OLMoConfigurationError(
                f"the SDE parametrization expects an AttentionConfig sequence mixer, got "
                f"{type(attention).__name__}"
            )
        if attention.n_heads != self.n_heads or (attention.head_dim or 0) != self.head_dim:
            raise OLMoConfigurationError(
                f"model config (n_heads={attention.n_heads}, head_dim={attention.head_dim}) does "
                f"not match this scaling (n_heads={self.n_heads}, head_dim={self.head_dim}). Note "
                f"that 'head_dim' must be set explicitly -- letting it default to "
                f"'d_model // n_heads' would re-couple the head count to the residual width."
            )

        feed_forward = config.block.feed_forward
        if feed_forward is None or feed_forward.hidden_size != self.hidden_size:
            raise OLMoConfigurationError(
                f"model config feed-forward hidden size does not match this scaling "
                f"(hidden_size={self.hidden_size})"
            )

        config.init_method = InitMethod.complete_p_sde

        attention.d_model_base = self.d_model_base
        attention.n_heads_base = self.n_heads_base
        attention.head_dim_base = self.head_dim
        attention.apply_qkv_mult = True

        config.block.feed_forward = FeedForwardConfig(
            name=FeedForwardType.complete_p,
            hidden_size=feed_forward.hidden_size,
            bias=feed_forward.bias,
            dtype=feed_forward.dtype,
            activation=feed_forward.activation,
            d_model_base=self.d_model_base,
            hidden_size_base=self.hidden_size_base,
        )

        config.lm_head.d_model_base = self.d_model_base

        config.block.attention_residual_alpha = self.residual_alpha
        config.block.feed_forward_residual_alpha = self.residual_alpha

        log.info("Joint SDE parametrization: %s", self.summary())
        if self.is_base:
            log.info("Target model is the base model: all multipliers are 1.0.")

        return config

    def check_alphas(self, rtol: float = 0.05) -> None:
        """
        Warn if rounding has moved either alpha away from its base value by more than ``rtol``.
        Only meaningful when scaling along the SDE ray.
        """
        for name, alpha, alpha_base in (
            ("alpha_mlp", self.alpha_mlp, self.alpha_mlp_base),
            ("alpha_att", self.alpha_att, self.alpha_att_base),
        ):
            drift = abs(math.log(alpha / alpha_base))
            if drift > math.log1p(rtol):
                log.warning(
                    "%s = %.4g has drifted %.1f%% from its base value %.4g. If you intended to "
                    "scale along the SDE ray, adjust the multipliers so that M and H land on "
                    "round numbers.",
                    name,
                    alpha,
                    100 * (alpha / alpha_base - 1),
                    alpha_base,
                )
