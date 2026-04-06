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
import math
from typing import List

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
from olmo_core.nn.transformer import TransformerConfig
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


def _hidden_size_base(d_model_base: int, multiple_of: int = 256) -> int:
    """Compute the FFN hidden size for the base model (mirrors llama_like logic)."""
    h = int(8 * d_model_base / 3)
    return multiple_of * math.ceil(h / multiple_of)


HIDDEN_SIZE_BASE = _hidden_size_base(D_MODEL_BASE)
# ───────────────────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 1024
GLOBAL_BATCH_SIZE = 576 * SEQUENCE_LENGTH  # ~524K tokens/step (divisible by rank_microbatch=64*1024 with any DP world size)
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
    parser.add_argument("--d-model", type=int, default=None, help="Model width (overrides MODEL_SIZE default).")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers (overrides MODEL_SIZE default).")
    parser.add_argument("--n-heads", type=int, default=None, help="Number of attention heads (overrides MODEL_SIZE default).")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate (overrides LR default).")
    parser.add_argument("--total-tokens", type=float, default=None, help="Total training tokens, e.g. 1e11 (overrides TOTAL_TOKENS default).")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Global batch size in tokens (overrides GLOBAL_BATCH_SIZE default).")
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant", "polynomial", "wsd"],
        help="Learning rate schedule: 'cosine' (default, CosWithWarmup), 'constant' (ConstantWithWarmup), 'polynomial' (LinearWithWarmup), or 'wsd' (WSD warmup-stable-decay, 10%% annealing by default).",
    )
    parser.add_argument(
        "--wsd-decay-fraction",
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


def _fmt_tokens(n: float) -> str:
    """Format a token count as a short string, e.g. 100B or 10B."""
    if n >= 1e12:
        return f"{n/1e12:.4g}T"
    if n >= 1e9:
        return f"{n/1e9:.4g}B"
    return f"{n/1e6:.4g}M"


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    # ── Resolve hyperparams (CLI args take priority over script-level constants) ──
    d_model      = opts.d_model       or D_MODEL
    n_layers     = opts.n_layers      or N_LAYERS
    n_heads      = opts.n_heads       or N_HEADS
    lr           = opts.lr            or LR
    total_tokens = int(opts.total_tokens or TOTAL_TOKENS)
    batch_size   = opts.global_batch_size or GLOBAL_BATCH_SIZE
    schedule          = opts.schedule  # "cosine" | "constant" | "polynomial" | "wsd"
    wsd_decay_fraction = opts.wsd_decay_fraction

    head_dim = d_model // n_heads

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
    optim_tag = "skip_adamw"
    run_name = (
        f"h{n_heads}_hd{head_dim}_L{n_layers}"
        f"_T{_fmt_tokens(total_tokens)}"
        f"_bs{batch_size // 1024}k"
        f"_lr{_fmt_lr(lr)}_{optim_tag}"
    )
    if COMPLETE_P:
        run_name += "_completep"
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
        block_name="reordered_norm",
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        attn_backend=AttentionBackendName.flash_2,
    )

    if COMPLETE_P:
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

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


if __name__ == "__main__":
    main(build_config, parser=_make_parser())
