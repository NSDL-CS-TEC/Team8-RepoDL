"""
Pipeline figure for the report and slides: figures/pipeline.png and figures/pipeline.pdf.
It only draws boxes; it reads no data.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

INK, MUTED = "#1f2933", "#52606d"
C_DATA, C_A, C_B, C_EVAL = "#e4ecf7", "#fde8d7", "#dcf0e3", "#ece6f6"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=color, ec=INK, lw=0.9))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9.5,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.095, body, ha="center", va="top", fontsize=7.8,
            color=MUTED, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color=INK, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(-0.02, 1.63)
    ax.set_ylim(0.0, 0.67)
    ax.axis("off")

    box(ax, 0.00, 0.20, 0.27, 0.30, "Driving EEG",
        "Cao et al. 2019\n26 subjects, 61 sessions\n32 ch, 500 Hz\nlane-deviation events",
        C_DATA)
    box(ax, 0.32, 0.20, 0.30, 0.30, "Pre-event window",
        "Fz, Pz, Oz; 1-50 Hz; 250 Hz\n3 s ending at deviation onset\nz-score per session\n"
        "label: RT vs session p5\nfast < 1.5x, slow > 2.5x", C_DATA)

    box(ax, 0.68, 0.39, 0.26, 0.25, "Raw signal", "tensor 1 x 3 x 750", C_A)
    box(ax, 0.68, 0.04, 0.26, 0.25, "CWT scalogram",
        "Morlet, 6 cycles, 1-40 Hz\nlog power, 3 x 128 x 128", C_B)
    box(ax, 0.99, 0.39, 0.25, 0.25, "EEGNet v4",
        "from scratch\n2,265 parameters\nlr 3e-3", C_A)
    box(ax, 0.99, 0.04, 0.25, 0.25, "ResNet18",
        "ImageNet init, full fine-tune\n11.2 M parameters\nlr 1e-4", C_B)
    box(ax, 1.30, 0.16, 0.30, 0.36, "Evaluation",
        "p(slow), class-weighted BCE\n5-fold CV by subject x 3 seeds\n"
        "balanced accuracy per subject\nΔ = ResNet18 - EEGNet\n95% bootstrap over subjects",
        C_EVAL)

    arrow(ax, 0.27, 0.35, 0.32, 0.35)
    arrow(ax, 0.62, 0.38, 0.68, 0.51)
    arrow(ax, 0.62, 0.32, 0.68, 0.17)
    arrow(ax, 0.94, 0.515, 0.99, 0.515)
    arrow(ax, 0.94, 0.165, 0.99, 0.165)
    arrow(ax, 1.24, 0.515, 1.30, 0.42)
    arrow(ax, 1.24, 0.165, 1.30, 0.26)

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "pipeline.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pipeline.pdf", bbox_inches="tight")
    print("Saved figures/pipeline.png and figures/pipeline.pdf")


if __name__ == "__main__":
    main()
