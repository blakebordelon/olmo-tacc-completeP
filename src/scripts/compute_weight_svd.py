#!/usr/bin/env python3
"""
Compute singular values of key linear layers at the initial and final checkpoints
of a training run and save results for later plotting.

The script auto-detects the number of layers from the checkpoint metadata.
Results are saved as a .pt file containing numpy arrays keyed by layer name.

Usage:
    # Basic: compare step0 (init) and the last permanent checkpoint (final)
    python src/scripts/compute_weight_svd.py \\
        --checkpoint-dir /path/to/save_folder/run_name \\
        --output svd_results.pt

    # Specify explicit steps
    python src/scripts/compute_weight_svd.py \\
        --checkpoint-dir /path/to/save_folder/run_name \\
        --initial-step 0 --final-step 50000

Loading the results:
    import torch, numpy as np
    r = torch.load("svd_results.pt", weights_only=False)
    # r["initial"] and r["final"] are dicts: layer_key -> np.ndarray of singular values
    sv_init = r["initial"]["model.blocks.0.attention.w_q.weight"]
    sv_final = r["final"]["model.blocks.0.attention.w_q.weight"]
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

# Plotting is optional — only imported when --plot is requested.
_PLOT_IMPORTS_OK = False


def _ensure_plot_imports() -> None:
    global _PLOT_IMPORTS_OK
    if _PLOT_IMPORTS_OK:
        return
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "matplotlib and seaborn are required for --plot.\n"
            "Install them with:  pip install matplotlib seaborn"
        ) from e
    _PLOT_IMPORTS_OK = True


def find_step_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """
    Return a {step: checkpoint_path} dict for all valid step checkpoints found
    in the run directory. A valid checkpoint has both a ``.metadata.json`` and a
    ``model_and_optim/`` subdirectory.
    """
    pattern = re.compile(r"^step(\d+)$")
    steps: Dict[int, Path] = {}
    for p in sorted(run_dir.iterdir()):
        m = pattern.match(p.name)
        if m and (p / ".metadata.json").exists() and (p / "model_and_optim").exists():
            steps[int(m.group(1))] = p
    return steps


def detect_n_layers(checkpoint_dir: Path) -> int:
    """Infer the number of transformer layers from checkpoint metadata keys."""
    meta = get_checkpoint_metadata(checkpoint_dir / "model_and_optim")
    indices = set()
    for k in meta.state_dict_metadata:
        m = re.search(r"model\.blocks\.(\d+)\.", k)
        if m:
            indices.add(int(m.group(1)))
    if not indices:
        raise ValueError(
            "Could not auto-detect n_layers from checkpoint metadata. "
            "Pass --n-layers explicitly."
        )
    return max(indices) + 1


def linear_layer_keys(n_layers: int) -> List[str]:
    """
    Return model state-dict keys for all key 2-D linear weight matrices:
    attention (w_q, w_k, w_v, w_out) and feed-forward (w1, w2, w3) for every
    layer, plus the embedding matrix and the LM-head projection.
    """
    keys: List[str] = []
    for i in range(n_layers):
        b = f"model.blocks.{i}"
        keys += [
            f"{b}.attention.w_q.weight",
            f"{b}.attention.w_k.weight",
            f"{b}.attention.w_v.weight",
            f"{b}.attention.w_out.weight",
            f"{b}.feed_forward.w1.weight",
            f"{b}.feed_forward.w2.weight",
            f"{b}.feed_forward.w3.weight",
        ]
    keys += [
        "model.embeddings.weight",
        "model.lm_head.w_out.weight",
    ]
    return keys


def svd_from_checkpoint(
    checkpoint_dir: Path,
    keys: List[str],
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load each weight tensor listed in *keys* from the checkpoint at
    *checkpoint_dir* and return a dict mapping key -> singular values (numpy
    array, float32, descending order).

    Weights are cast to float32 before SVD for numerical stability.
    """
    model_and_optim = checkpoint_dir / "model_and_optim"

    # Filter to keys that actually exist in this checkpoint.
    meta = get_checkpoint_metadata(model_and_optim)
    available = set(meta.state_dict_metadata.keys())
    missing = [k for k in keys if k not in available]
    if missing:
        print(f"  [warn] {len(missing)} keys not found in checkpoint, skipping:")
        for k in missing:
            print(f"         {k}")
    keys_to_load = [k for k in keys if k in available]

    results: Dict[str, np.ndarray] = {}
    tensors = list(load_keys(model_and_optim, keys_to_load))
    for key, tensor in zip(keys_to_load, tensors):
        w = tensor.float()  # cast to fp32
        if w.ndim == 1:
            # Shouldn't happen for linear weights, but handle gracefully.
            results[key] = w.abs().numpy()
        else:
            # Flatten any extra dims (e.g. embedding: vocab x d_model is already 2D).
            sv = torch.linalg.svdvals(w.reshape(w.shape[0], -1))
            results[key] = sv.numpy()
        if verbose:
            print(f"  {key:60s}  shape={list(w.shape)}  rank={len(results[key])}")
    return results


_COMPONENT_TYPES = ["w_q", "w_k", "w_v", "w_out", "w1", "w2", "w3"]


def plot_svd_distributions(
    svd_initial: Dict[str, np.ndarray],
    svd_final: Dict[str, np.ndarray],
    initial_step: int,
    final_step: int,
    plot_output: Optional[Path],
) -> None:
    """
    For each component type (w_q, w_k, …) produce one figure with one subplot
    per layer.  Each subplot shows a seaborn KDE of the singular-value
    distribution at the initial checkpoint (blue) and the final checkpoint
    (orange).

    :param svd_initial: Dict mapping state-dict key -> singular values at init.
    :param svd_final: Dict mapping state-dict key -> singular values at final.
    :param initial_step: Step number of the initial checkpoint (for labels).
    :param final_step: Step number of the final checkpoint (for labels).
    :param plot_output: Directory to write PNG files.  If ``None``, calls
        ``plt.show()`` instead.
    """
    _ensure_plot_imports()
    import matplotlib.pyplot as plt
    import seaborn as sns

    rocket = sns.color_palette("rocket", as_cmap=True)
    color_init = rocket(0.25)
    color_final = rocket(0.7)

    # Re-index as  component -> {layer_idx: sv_array}
    def _group(sv_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[int, np.ndarray]]:
        grouped: Dict[str, Dict[int, np.ndarray]] = {}
        for key, sv in sv_dict.items():
            m = re.match(r"model\.blocks\.(\d+)\..*?\.(w_\w+)\.weight", key)
            if m:
                layer, comp = int(m.group(1)), m.group(2)
                grouped.setdefault(comp, {})[layer] = sv
        return grouped

    grouped_init = _group(svd_initial)
    grouped_final = _group(svd_final)

    for comp in _COMPONENT_TYPES:
        layers_init = grouped_init.get(comp, {})
        layers_final = grouped_final.get(comp, {})
        layer_indices = sorted(set(layers_init) | set(layers_final))
        if not layer_indices:
            continue

        n = len(layer_indices)
        ncols = min(8, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 2.8 * nrows), squeeze=False
        )
        fig.suptitle(f"Singular-value distributions — {comp}", fontsize=13, y=1.01)

        for idx, layer in enumerate(layer_indices):
            ax = axes[idx // ncols][idx % ncols]
            sv_i = layers_init.get(layer)
            sv_f = layers_final.get(layer)
            if sv_i is not None:
                sns.kdeplot(sv_i, ax=ax, color=color_init, fill=True, alpha=0.3,
                            label=f"step {initial_step}", linewidth=1.2)
            if sv_f is not None:
                sns.kdeplot(sv_f, ax=ax, color=color_final, fill=True, alpha=0.3,
                            label=f"step {final_step}", linewidth=1.2)
            ax.set_title(f"layer {layer}", fontsize=8)
            ax.set_xlabel("σ", fontsize=7)
            ax.set_ylabel("density", fontsize=7)
            ax.tick_params(labelsize=6)

        # Hide unused subplots.
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        # One shared legend in the first subplot.
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=9)

        plt.tight_layout()

        if plot_output is not None:
            plot_output.mkdir(parents=True, exist_ok=True)
            path = plot_output / f"{comp}_sv_distributions.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  saved {path}")
        else:
            plt.show()
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save singular values of linear layers at init and final checkpoints."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Run directory containing step0/, step1000/, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt file. Defaults to <checkpoint-dir>/svd_results.pt",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of transformer layers. Auto-detected from checkpoint if not given.",
    )
    parser.add_argument(
        "--initial-step",
        type=int,
        default=None,
        help="Step number to use as the 'initial' checkpoint (default: smallest found).",
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=None,
        help="Step number to use as the 'final' checkpoint (default: largest found).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="After computing SVDs, plot per-layer singular-value KDE distributions. "
        "Requires matplotlib and seaborn.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Directory to save plot PNGs. Defaults to <output>_plots/ next to the "
        ".pt file. Pass 'show' to display interactively instead of saving.",
    )
    args = parser.parse_args()

    run_dir: Path = args.checkpoint_dir
    output: Path = args.output or (run_dir / "svd_results.pt")

    # ── Discover checkpoints ──────────────────────────────────────────────────
    step_map = find_step_checkpoints(run_dir)
    if not step_map:
        raise FileNotFoundError(f"No valid step checkpoints found in {run_dir}")

    all_steps = sorted(step_map)
    print(f"Found checkpoints at steps: {all_steps}")

    initial_step: int = args.initial_step if args.initial_step is not None else all_steps[0]
    final_step: int = args.final_step if args.final_step is not None else all_steps[-1]

    for step, label in [(initial_step, "initial"), (final_step, "final")]:
        if step not in step_map:
            raise FileNotFoundError(
                f"{label} step {step} not found. Available steps: {all_steps}"
            )

    print(f"Initial checkpoint : step {initial_step}  ({step_map[initial_step]})")
    print(f"Final   checkpoint : step {final_step}  ({step_map[final_step]})")

    # ── Detect n_layers ───────────────────────────────────────────────────────
    n_layers: int = args.n_layers or detect_n_layers(step_map[initial_step])
    print(f"n_layers = {n_layers}")

    keys = linear_layer_keys(n_layers)
    print(f"\nComputing SVD for {len(keys)} weight matrices...")

    # ── Compute SVDs ──────────────────────────────────────────────────────────
    print(f"\n[step {initial_step}]")
    svd_initial = svd_from_checkpoint(step_map[initial_step], keys)

    print(f"\n[step {final_step}]")
    svd_final = svd_from_checkpoint(step_map[final_step], keys)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "initial_step": initial_step,
        "final_step": final_step,
        "n_layers": n_layers,
        "keys": keys,
        "initial": svd_initial,
        "final": svd_final,
    }
    torch.save(results, output)
    print(f"\nSaved to {output}")
    print(
        "\nTo load:\n"
        "  import torch\n"
        f"  r = torch.load('{output}', weights_only=False)\n"
        "  sv_init  = r['initial']['model.blocks.0.attention.w_q.weight']\n"
        "  sv_final = r['final']['model.blocks.0.attention.w_q.weight']"
    )

    if args.plot:
        plot_output: Optional[Path]
        if args.plot_output is not None and str(args.plot_output) == "show":
            plot_output = None  # interactive display
        else:
            plot_output = args.plot_output or output.parent / (output.stem + "_plots")

        print("\nGenerating singular-value distribution plots …")
        plot_svd_distributions(
            svd_initial,
            svd_final,
            initial_step=initial_step,
            final_step=final_step,
            plot_output=plot_output,
        )


if __name__ == "__main__":
    main()
