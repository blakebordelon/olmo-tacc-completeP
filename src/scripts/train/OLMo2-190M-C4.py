"""
Pre-training script for OLMo2-style models on C4 data with GPT2 tokenizer.

Control model size by setting MODEL_SIZE below (or pass CLI sweep args to override individual dims.

Usage:
    # Dry run (print config and exit):
    python src/scripts/train/OLMo2-190M-C4.py --dry-run --save-folder=/tmp/test

    # Single-GPU training:
    python src/scripts/train/OLMo2-190M-C4.py --train-single --save-folder=/path/to/checkpoints

    # Multi-GPU training with torchrun:
    torchrun --nproc-per-node=8 src/scripts/train/OLMo2-190M-C4.py --save-folder=/path/to/checkpoints

    # Override individual config fields (dot notation):
    python src/scripts/train/OLMo2-190M-C4.py --dry-run --save-folder=/tmp/test model.init_std=0.01

    # Sweep over model dimensions (run name and save subdirectory are auto-generated):
    for n_layers in 4 8 12; do
      torchrun --nproc-per-node=3 src/scripts/train/OLMo2-190M-C4.py \\
        --save-folder=/scratch/.../experiments --n-layers=$n_layers &
    done
"""

import argparse
import logging
import math
from dataclasses import replace
from typing import List, Optional

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig, FeedForwardType
from olmo_core.nn.transformer import JointScalingConfig, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod
from olmo_core.optim import (
    ConstantWithWarmup,
    CosWithWarmup,
    LinearWithWarmup,
    OptimGroupOverride,
    SkipStepAdamWConfig,
    WSD,
)
from olmo_core.script_utils import ExperimentConfig, get_cli_parser, main
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)


def _log_attn_backend(model_config: TransformerConfig):
    """
    Log which attention backend the model is configured to use, and whether flash-attn is
    actually importable on this node. Purely diagnostic: throughput silently collapses if a
    run falls back to the unfused SDPA math path, so we want it stated plainly in the logs.

    :param model_config: The (already built) transformer config to inspect.
    """
    from olmo_core.nn.attention.flash_attn_api import has_flash_attn_2

    backend = getattr(model_config.block.attention, "backend", None)
    available = has_flash_attn_2()
    try:
        import flash_attn

        version = flash_attn.__version__
    except ImportError:
        version = "not installed"

    log.info(f"ATTENTION BACKEND: requested={backend}, flash_attn_2_available={available} (flash-attn {version})")
    if not available:
        log.warning(
            "flash-attn is NOT available on this node. If the backend above is a flash_* variant "
            "the run will fail at model build; if it is 'torch' you are on the slow SDPA math path."
        )

# ── Model size selection ────────────────────────────────────────────────────────
# Change MODEL_SIZE to quickly switch between preset architectures, or set
# D_MODEL / N_LAYERS / N_HEADS directly below for a custom size.
#
#  Name   │  d_model │ n_layers │ n_heads │ approx params
#  ────────┼──────────┼──────────┼─────────┼──────────────
#  "14M"   │     128  │       4  │       4  │ ~14M
#  "85M"   │     512  │       8  │       8  │ ~85M
#  "190M"  │     768  │      12  │      12  │ ~190M
#  "370M"  │    1024  │      16  │      16  │ ~370M
#  "760M"  │    1280  │      20  │      16  │ ~760M
#  "1B"    │    2048  │      16  │      16  │ ~1B
MODEL_SIZES = {
    "14M":  dict(d_model=128,  n_layers=4,  n_heads=4),
    "85M":  dict(d_model=512,  n_layers=8,  n_heads=8),
    "190M": dict(d_model=768,  n_layers=12, n_heads=12),
    "370M": dict(d_model=1024, n_layers=16, n_heads=16),
    "760M": dict(d_model=1280, n_layers=20, n_heads=16),
    "1B":   dict(d_model=2048, n_layers=16, n_heads=16),
}

MODEL_SIZE = "190M"  # ← change this to switch sizes

D_MODEL  = MODEL_SIZES[MODEL_SIZE]["d_model"]
N_LAYERS = MODEL_SIZES[MODEL_SIZE]["n_layers"]
N_HEADS  = MODEL_SIZES[MODEL_SIZE]["n_heads"]
# ───────────────────────────────────────────────────────────────────────────────

# ── CompleteP parametrization ───────────────────────────────────────────────────
# Set COMPLETE_P = True to enable the CompleteP (μP-like) parametrization.
# D_MODEL_BASE / N_LAYERS_BASE define the reference model; all scaling rules are
# derived relative to these.  When D_MODEL == D_MODEL_BASE and N_LAYERS ==
# N_LAYERS_BASE every multiplier is 1.0 and training is identical to standard.
#
# What CompleteP changes at (D_MODEL, N_LAYERS) ≠ (D_MODEL_BASE, N_LAYERS_BASE):
#   - Attention softmax scale:    sqrt(d_head_base) / d_head  (equals 1/sqrt(d_head_base) at base size)
#   - Attention w_out multiplier: d_model_base / d_model
#   - FFN w1/w3 multiplier:       d_model_base / d_model
#   - FFN w2   multiplier:        hidden_size_base / hidden_size
#   - Residual update (depth):    x = x + (n_layers_base / n_layers) * BLOCK(x)
#   - All weight init stds:       std * sqrt(d_model / d_model_base)
COMPLETE_P = True  # ← set True to use CompleteP
D_MODEL_BASE  = MODEL_SIZES["190M"]["d_model"]   # 768  — reference width
N_LAYERS_BASE = MODEL_SIZES["190M"]["n_layers"]  # 12   — reference depth
N_HEADS_BASE  = MODEL_SIZES["190M"]["n_heads"]   # 12   — reference head count


def _hidden_size_base(d_model_base: int, multiple_of: int = 256) -> int:
    """Compute the FFN hidden size for the base model (mirrors llama_like logic)."""
    h = int(8 * d_model_base / 3)
    return multiple_of * math.ceil(h / multiple_of)


HIDDEN_SIZE_BASE = _hidden_size_base(D_MODEL_BASE)
# ───────────────────────────────────────────────────────────────────────────────

# ── Joint SDE parametrization (--param=sde) ────────────────────────────────────
# Relaxes the joint scaling of the residual width N, the depth L, and the block hidden widths
# (MLP hidden size M, attention head count H).  Down projections (FFN w2, attention w_out) are
# initialized with std ∝ sqrt(N) rather than sqrt(fan-in), which -- because Adam's update is
# entrywise scale-invariant -- opens a gap between a block's incoherent (init) and coherent
# (Adam) responses:
#
#   init:  each block writes O(sqrt(alpha / L)) into the residual stream → sums diffusively to O(sqrt(alpha))
#   Adam:  each block's contribution moves by O(eta / L) per step        → sums coherently  to O(eta)
#
# Both are O(1) with a learning rate constant in N, M, H and L.  The diffusion coefficients are
#
#   alpha_mlp = N / (M * L)      alpha_att = N / (H * d_head * L)
#
# --scaling=sde   holds both alphas fixed  (N ×w, L ×d  ⇒  M ×w/d, H ×w/d)
# --scaling=linear scales N ~ M ~ H ~ L ~ s, sending both alphas to 0 like 1/s (ODE limit)
#
# HEAD_DIM is held FIXED across the sweep: it is the head *count* that scales, so that the fan-in
# of w_out is proportional to H.  Requires the pre-norm block (block_name="default"): the
# reordered_norm block's RMSNorm sits on the branch *output* and, being 0-homogeneous, would
# cancel the sqrt(N) init exactly.
HEAD_DIM = 64
# ───────────────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 1024
GLOBAL_BATCH_SIZE = 576 * SEQUENCE_LENGTH  # ~524K tokens/step (divisible by rank_microbatch=32*1024 with any DP world size)
LR = 1e-3
TOTAL_TOKENS = int(4e9)  # 4B tokens

TRAIN_DATA_PATHS = [
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-000-00000.npy",
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-001-00000.npy",
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-002-00000.npy",
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-003-00000.npy",
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-004-00000.npy",
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-005-00000.npy",    
]
VAL_DATA_PATHS = [
    "/scratch/11423/blakebordelon/ICL_proj/datasets/olmo/part-0-00000.npy"
]


def _make_parser() -> argparse.ArgumentParser:
    """Extend the standard CLI parser with sweep-friendly per-run overrides."""
    parser = get_cli_parser()
    parser.add_argument("--d-model", "--d_model", type=int, default=None, help="Model width (overrides MODEL_SIZE default).")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=None, help="Number of transformer layers (overrides MODEL_SIZE default).")
    parser.add_argument("--n-heads", "--n_heads", type=int, default=None, help="Number of attention heads (overrides MODEL_SIZE default).")
    parser.add_argument("--head-dim", "--head_dim", type=int, default=None, help="Attention head dim. Only used with --param=sde, where it is held fixed across the sweep (default: HEAD_DIM).")
    parser.add_argument("--hidden-size", "--hidden_size", type=int, default=None, help="FFN hidden size M. Only used with --param=sde (default: derived from the scaling recipe).")
    parser.add_argument(
        "--param",
        type=str,
        default="complete_p" if COMPLETE_P else "standard",
        choices=["standard", "complete_p", "sde"],
        help="Parametrization: 'standard', 'complete_p' (CompleteP, ODE depth limit), or 'sde' (joint SDE scaling with sqrt(N) down-projection init).",
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="sde",
        choices=["sde", "linear"],
        help="With --param=sde: 'sde' holds alpha_mlp and alpha_att fixed; 'linear' scales N ~ M ~ H ~ L together (alphas fall like 1/mult).",
    )
    parser.add_argument(
        "--block",
        type=str,
        default=None,
        choices=["default", "reordered_norm"],
        help="Transformer block type. Defaults to 'default' (pre-norm) for --param=sde and 'reordered_norm' otherwise. Use --param=complete_p --block=default as the control run that isolates the init change from the block change.",
    )
    parser.add_argument("--width-mult", "--width_mult", type=float, default=1.0, help="With --param=sde --scaling=sde: multiplier on d_model relative to the base model.")
    parser.add_argument("--depth-mult", "--depth_mult", type=float, default=1.0, help="With --param=sde --scaling=sde: multiplier on n_layers relative to the base model.")
    parser.add_argument("--linear-mult", "--linear_mult", type=float, default=1.0, help="With --param=sde --scaling=linear: multiplier on all of N, M, H, L.")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate (overrides LR default).")
    parser.add_argument("--total-tokens", "--total_tokens", type=float, default=None, help="Total training tokens, e.g. 1e11 (overrides TOTAL_TOKENS default).")
    parser.add_argument("--global-batch-size", "--global_batch_size", type=int, default=None, help="Global batch size in tokens (overrides GLOBAL_BATCH_SIZE default).")
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant", "polynomial", "wsd"],
        help="Learning rate schedule: 'cosine' (default, CosWithWarmup), 'constant' (ConstantWithWarmup), 'polynomial' (LinearWithWarmup), or 'wsd' (WSD warmup-stable-decay, 10%% annealing by default).",
    )
    parser.add_argument(
        "--wsd-decay-fraction",
        "--wsd_decay_fraction",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear decay in the WSD schedule (default: 0.1).",
    )
    return parser


def _fmt_lr(lr: float) -> str:
    """Format a learning rate as a short string, e.g. 3e-4."""
    s = f"{lr:.0e}"          # '3e-04'
    s = s.replace("e-0", "e-").replace("e+0", "e")  # '3e-4'
    return s


def _fmt_mult(x: float) -> str:
    """Format a scaling multiplier as a short string, e.g. 2, 1.5, 0.5."""
    return f"{x:.3g}"


def _fmt_tokens(n: float) -> str:
    """Format a token count as a short string, e.g. 100B or 10B."""
    if n >= 1e12:
        return f"{n/1e12:.4g}T"
    if n >= 1e9:
        return f"{n/1e9:.4g}B"
    return f"{n/1e6:.4g}M"


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # ── Resolve hyperparams (CLI args take priority over script-level constants) ──
    lr           = opts.lr            or LR
    total_tokens = int(opts.total_tokens or TOTAL_TOKENS)
    batch_size   = opts.global_batch_size or GLOBAL_BATCH_SIZE
    schedule          = opts.schedule  # "cosine" | "constant" | "polynomial" | "wsd"
    wsd_decay_fraction = opts.wsd_decay_fraction
    param             = opts.param     # "standard" | "complete_p" | "sde"

    # ── Resolve model dimensions ─────────────────────────────────────────────────
    scaling: Optional[JointScalingConfig] = None
    hidden_size: Optional[int] = None

    if param == "sde":
        # N, M, H, L come from the scaling recipe, with HEAD_DIM held fixed.
        base = JointScalingConfig.base(
            d_model=D_MODEL_BASE,
            n_layers=N_LAYERS_BASE,
            n_heads=N_HEADS_BASE,
            head_dim=opts.head_dim or HEAD_DIM,
            hidden_size=HIDDEN_SIZE_BASE,
        )
        if opts.scaling == "sde":
            scaling = base.scale_sde(width_mult=opts.width_mult, depth_mult=opts.depth_mult)
        else:
            scaling = base.scale_linear(mult=opts.linear_mult)

        # Explicit dimension args override the recipe, for one-off points off the ray.
        scaling = replace(
            scaling,
            d_model=opts.d_model or scaling.d_model,
            n_layers=opts.n_layers or scaling.n_layers,
            n_heads=opts.n_heads or scaling.n_heads,
            hidden_size=opts.hidden_size or scaling.hidden_size,
        )
        scaling.check_alphas()

        d_model, n_layers, n_heads = scaling.d_model, scaling.n_layers, scaling.n_heads
        head_dim, hidden_size = scaling.head_dim, scaling.hidden_size
    else:
        d_model  = opts.d_model  or D_MODEL
        n_layers = opts.n_layers or N_LAYERS
        n_heads  = opts.n_heads  or N_HEADS
        head_dim = d_model // n_heads

    block_name = opts.block or ("default" if scaling is not None else "reordered_norm")
    if scaling is not None and block_name != "default":
        raise ValueError(
            "--param=sde requires --block=default: a block that normalizes the residual branch "
            "output cancels the sqrt(N) init on the down projections exactly."
        )

    # ── Build scheduler ──────────────────────────────────────────────────────────
    if schedule == "cosine":
        scheduler = CosWithWarmup(warmup_steps=2000)
    elif schedule == "constant":
        scheduler = ConstantWithWarmup(warmup=2000)
    elif schedule == "polynomial":
        scheduler = LinearWithWarmup(warmup=2000)
    elif schedule == "wsd":
        scheduler = WSD(warmup=2000, decay_fraction=wsd_decay_fraction)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    # ── Auto-generate run name from hyperparams ──────────────────────────────────
    # `family` leads the run name and doubles as the W&B group. The two rays of --param=sde share
    # a parametrization (complete_p_sde init) and differ only in how the block hidden widths scale
    # with N and L, which is exactly what the diffusion coefficients alpha_mlp / alpha_att see:
    #   sdescale  holds both alphas fixed        -> diffusion survives, SDE depth limit
    #   odescale  scales N ~ M ~ H ~ L linearly  -> alphas fall like 1/s, drift-only ODE limit
    # ('odescale' is *not* 'completep': same ODE limit, different parametrization at finite size.)
    if scaling is not None:
        family = "sdescale" if opts.scaling == "sde" else "odescale"
    elif param == "complete_p":
        family = "completep"
    else:
        family = "standard"

    optim_tag = "skip_adamw"
    run_name = (
        f"h{n_heads}_hd{head_dim}_L{n_layers}"
        f"_T{_fmt_tokens(total_tokens)}"
        f"_bs{batch_size // 1024}k"
        f"_lr{_fmt_lr(lr)}_{optim_tag}"
    )
    if scaling is not None:
        # Multipliers are read back off the *final* dims rather than the CLI args, so they stay
        # honest when an individual dim is overridden off the ray. The alphas are derivable from
        # N/M/H/L and constant along the SDE ray by construction, so they live in the W&B config
        # rather than eating characters in every name.
        mult_tag = (
            f"s{_fmt_mult(scaling.width_mult)}"
            if opts.scaling == "linear"
            else f"w{_fmt_mult(scaling.width_mult)}_d{_fmt_mult(scaling.depth_mult)}"
        )
        run_name = (
            f"{family}_{mult_tag}"
            f"_N{d_model}_M{hidden_size}_H{n_heads}_hd{head_dim}_L{n_layers}"
            f"_T{_fmt_tokens(total_tokens)}"
            f"_bs{batch_size // 1024}k"
            f"_lr{_fmt_lr(lr)}_{optim_tag}"
        )
    elif param == "complete_p":
        run_name += "_completep"
    if scaling is None and block_name != "reordered_norm":
        run_name += f"_{block_name}"
    if schedule == "wsd":
        run_name += f"_wsd{int(wsd_decay_fraction * 100)}pct"
    elif schedule != "cosine":
        run_name += f"_{schedule}"
    # opts.name overrides the auto-generated name (useful for one-off runs)
    run_name = opts.name or run_name
    save_folder = f"{opts.save_folder}/{run_name}"

    sequence_length = opts.sequence_length or SEQUENCE_LENGTH
    tokenizer_config = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    model_config = TransformerConfig.llama_like(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        vocab_size=tokenizer_config.padded_vocab_size(),
        # The SDE parametrization needs the pre-norm block: reordered_norm puts an RMSNorm on the
        # residual branch *output*, which (being 0-homogeneous) cancels the sqrt(N) init on the
        # down projections exactly and would silently reduce the model to CompleteP.
        block_name=block_name,
        # Pin head_dim so the head count H scales independently of the residual width N.
        head_dim=head_dim if scaling is not None else None,
        feed_forward=(
            FeedForwardConfig(hidden_size=hidden_size, bias=False, dtype=DType.float32)
            if hidden_size is not None
            else None
        ),
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        attn_backend=AttentionBackendName.flash_2,
    )

    if scaling is not None:
        # Sets init_method=complete_p_sde, the CompleteP feed-forward, the attention/FFN/LM-head
        # base dims (d_model_base, n_heads_base, head_dim_base, hidden_size_base), the QKV
        # multiplier, and residual_alpha = L_base / L on both branches.
        scaling.apply(model_config)
    elif param == "complete_p":
        # Switch to CompleteP init (scales all weight stds by sqrt(d_model / d_model_base)).
        model_config.init_method = InitMethod.complete_p
        # Attention: store d_model_base so Attention.__init__ can set w_out_mult and
        # override softmax_scale to d_model_base / d_model.
        model_config.block.sequence_mixer.d_model_base = D_MODEL_BASE  # type: ignore[union-attr]
        # Feed-forward: switch to CompletePFeedForward which applies w1/w3/w2 multipliers.
        ff = model_config.block.feed_forward
        assert ff is not None, "Expected block.feed_forward to be set for CompleteP"
        model_config.block.feed_forward = FeedForwardConfig(
            name=FeedForwardType.complete_p,
            hidden_size=ff.hidden_size,
            bias=ff.bias,
            dtype=ff.dtype,
            activation=ff.activation,
            d_model_base=D_MODEL_BASE,
            hidden_size_base=HIDDEN_SIZE_BASE,
        )
        # LM head: apply d_model_base / d_model multiplier to logits. This keeps the effective
        # readout scale width-independent and implicitly scales the Adam LR on the readout weight
        # by d_model_base / d_model, matching the muP prescription. Init std is left at the
        # constant init_std (no width scaling) per the CompleteP readout rule.
        model_config.lm_head.d_model_base = D_MODEL_BASE  # type: ignore[union-attr]
        # Depth scaling: residual update becomes x = x + (n_layers_base / n_layers) * BLOCK(x).
        # This is implemented via the existing residual_alpha fields on TransformerBlockConfig,
        # which feed into ResidualStream.forward: torch.add(residual, block_out, alpha=alpha).
        depth_alpha = N_LAYERS_BASE / n_layers
        model_config.block.attention_residual_alpha = depth_alpha
        model_config.block.feed_forward_residual_alpha = depth_alpha

    dataset_config = NumpyFSLDatasetConfig(
        paths=TRAIN_DATA_PATHS,
        sequence_length=sequence_length,
        tokenizer=tokenizer_config,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=batch_size,
        seed=34521,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=32 * sequence_length,
        max_sequence_length=sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        scheduler=scheduler,
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.bfloat16,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )

    # Scalars to log as the W&B run config, so runs can be grouped/plotted by the scaling
    # dimensions. `main()` only pushes the full experiment config into ConfigSaverCallback, not
    # into WandBCallback, so without this the W&B run has no config at all.
    wandb_config = {
        "family": family,
        "param": param,
        "block": block_name,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size if hidden_size is not None else model_config.block.feed_forward.hidden_size,  # type: ignore[union-attr]
        "lr": lr,
        "schedule": schedule,
        "global_batch_size": batch_size,
        "sequence_length": sequence_length,
        "total_tokens": total_tokens,
    }
    # W&B tags/group: `family` gives a native grouping of the runs by parametrization + ray, so
    # the two rays of --param=sde can be filtered without regex-matching run names.
    wandb_tags = [f"param-{param}", f"block-{block_name}", family]
    if scaling is not None:
        wandb_tags.append(f"scaling-{opts.scaling}")
        wandb_config.update(
            scaling_mode=opts.scaling,
            width_mult=scaling.width_mult,
            depth_mult=scaling.depth_mult,
            hidden_mult=scaling.hidden_mult,
            head_mult=scaling.head_mult,
            alpha_mlp=scaling.alpha_mlp,
            alpha_att=scaling.alpha_att,
            alpha_mlp_base=scaling.alpha_mlp_base,
            alpha_att_base=scaling.alpha_att_base,
            residual_alpha=scaling.residual_alpha,
            d_model_base=scaling.d_model_base,
            n_layers_base=scaling.n_layers_base,
            n_heads_base=scaling.n_heads_base,
            hidden_size_base=scaling.hidden_size_base,
        )

    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(total_tokens),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: set to True to enable Comet logging
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=True,
                # Set the wandb entity where your project will be logged (generally your team name).
                entity="blake_bordelon",
                # Set the wandb project where this run will be logged.
                project="olmo_simple",
                group=family,
                tags=wandb_tags,
                config=wandb_config,
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=VAL_DATA_PATHS,
                    sequence_length=sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                    metadata=[{"label": "c4-validation"}],
                ),
                eval_interval=1000,
            ),
        )
    )

    _log_attn_backend(model_config)

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=_make_parser())
