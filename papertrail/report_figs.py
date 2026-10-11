"""Figures for the report.

    python -m papertrail.report_figs

Reads the per-epoch logs in runs/ and the metrics in results/, and writes PNG
and PDF files to results/figures/.
"""
import json

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt

from papertrail.config import REPO_ROOT

RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Colour follows the split, the same in every panel. The two hues stay distinct
# under colour-blindness, and the dashed training line still separates them in
# a greyscale print.
VALIDATION = "#2a78d6"
TRAINING = "#eb6834"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
})


def loss_curves(panels: list[tuple[str, str]], path, reference: tuple[str, float] | None = None) -> None:
    """Training and validation loss per epoch, one panel per run, on shared axes.

    panels:    (title, run_name) pairs; each run's log is runs/<run_name>/log.json.
    reference: an optional (label, loss) drawn as a horizontal line in every
               panel, e.g. the n-gram model's validation loss.

    The dot marks the epoch with the lowest validation loss, which is the
    checkpoint that was kept. Training loss is measured with dropout on, so it
    sits higher than it would on the same data in evaluation mode.
    """
    logs = [json.load(open(RUNS_DIR / run_name / "log.json")) for _, run_name in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.5), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    longest = max(len(log) for log in logs)

    for ax, (title, _), log in zip(axes, panels, logs):
        epochs = [entry["epoch"] for entry in log]
        train = [entry["train_loss"] for entry in log]
        val = [entry["val_loss"] for entry in log]

        if reference is not None:
            ax.axhline(reference[1], color=AXIS, linewidth=1, zorder=1.5)
        ax.plot(epochs, train, color=TRAINING, linewidth=2, linestyle=(0, (3, 1.5)), zorder=2)
        ax.plot(epochs, val, color=VALIDATION, linewidth=2, solid_capstyle="round", zorder=3)

        best = min(range(len(val)), key=val.__getitem__)
        ax.plot(epochs[best], val[best], "o", color=VALIDATION, markersize=6.5,
                markeredgecolor="white", markeredgewidth=1.5, zorder=4)
        # The label sits just above the flat stretch of the validation curve,
        # on whichever side of the dot has room, clear of the steep early epochs.
        room_on_right = epochs[best] <= longest / 2
        ax.annotate(f"best {val[best]:.2f} (epoch {epochs[best]})", (epochs[best], val[best]),
                    xytext=(3 if room_on_right else -3, 6), textcoords="offset points",
                    ha="left" if room_on_right else "right", va="bottom",
                    color=INK_SECONDARY, fontsize=7, zorder=5)

        ax.set_title(title, color=INK, fontsize=9, fontweight="bold", loc="left", pad=6)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("epoch", color=INK_SECONDARY)

    axes[0].set_xlim(0.5, longest + 0.5)
    axes[0].set_xticks([1] + list(range(5, longest + 1, 5)))
    axes[0].set_ylabel("loss (nats per token)", color=INK_SECONDARY)

    handles = [
        plt.Line2D([], [], color=VALIDATION, linewidth=2, label="validation"),
        plt.Line2D([], [], color=TRAINING, linewidth=2, linestyle=(0, (3, 1.5)), label="training (dropout on)"),
    ]
    if reference is not None:
        handles.append(plt.Line2D([], [], color=AXIS, linewidth=1, label=reference[0]))
    fig.legend(handles=handles, loc="upper right", ncols=len(handles), frameon=False, labelcolor=INK_SECONDARY,
               handlelength=2.6, columnspacing=1.6, borderaxespad=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=250, facecolor="white")
    plt.close(fig)
    print(f"wrote {path.with_suffix('.png').relative_to(REPO_ROOT)} and .pdf")


def main() -> None:
    ngram = json.load(open(RESULTS_DIR / "ngram_kn3.json"))
    loss_curves(
        [("RNN", "rnn"), ("LSTM", "lstm"), ("Transformer", "transformer")],
        FIGURES_DIR / "part1_loss_curves",
        reference=("KN trigram, validation", ngram["loss"]["validation"]),
    )


if __name__ == "__main__":
    main()
