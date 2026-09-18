"""Paper figures, read from results/tables/, written to results/figures/ at 300 dpi.

Greyscale-safe by construction: identity is carried by lightness, hatching
and marker shape -- never by hue -- so the figures survive a black-and-white
printer and every form of colour-vision deficiency. No chartjunk: recessive
grid, no top/right spines, direct value labels instead of a number on every
tick, and exactly one y-axis per figure.

Figure 3 (predicted vs actual, log-log) is the leak detector: if the points
hug the diagonal almost perfectly, the model has access to something it
should not have at arrival time.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TABLES_DIR = os.path.join("results", "tables")
FIGURES_DIR = os.path.join("results", "figures")
DPI = 300
SECONDS_PER_HOUR = 3600

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"
BAR_FILL = "#b8b8b8"
HIGHLIGHT_FILL = "#4a4a4a"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def figure_waiting_time_by_scheduler():
    comparison = pd.read_csv(os.path.join(TABLES_DIR, "scheduler_comparison.csv"))
    oracle_hours = comparison.loc[comparison["scheduler"] == "SRTF+Oracle", "avg_waiting_time"].iloc[0] / SECONDS_PER_HOUR
    bars = comparison.loc[comparison["scheduler"] != "SRTF+Oracle"].reset_index(drop=True)
    hours = bars["avg_waiting_time"] / SECONDS_PER_HOUR

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(bars))
    colors = [HIGHLIGHT_FILL if s == "SRTF+GBM" else BAR_FILL for s in bars["scheduler"]]
    hatches = ["////" if s == "SRTF+GBM" else "" for s in bars["scheduler"]]
    rects = ax.bar(x, hours, width=0.62, color=colors, edgecolor=INK, linewidth=0.6)
    for rect, hatch in zip(rects, hatches):
        rect.set_hatch(hatch)

    for rect, value in zip(rects, hours):
        ax.annotate(
            f"{value:.1f}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )

    ax.axhline(
        oracle_hours,
        color=INK,
        linestyle="--",
        linewidth=1.2,
        label=f"SRTF + Oracle (ceiling): {oracle_hours:.1f} h",
    )
    ax.legend(loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(bars["scheduler"], rotation=20, ha="right")
    ax.set_ylabel("Average waiting time (hours)")
    ax.set_ylim(0, hours.max() * 1.12)
    ax.set_title("Average waiting time by scheduler on the chronological test set", loc="left")
    _save(fig, "fig1_waiting_time_by_scheduler.png")


def figure_sensitivity():
    comparison = pd.read_csv(os.path.join(TABLES_DIR, "scheduler_comparison.csv"))
    rr_hours = comparison.loc[comparison["scheduler"] == "RR(q=4)", "avg_waiting_time"].iloc[0] / SECONDS_PER_HOUR
    uniform = pd.read_csv(os.path.join(TABLES_DIR, "sensitivity.csv"))
    lognormal = pd.read_csv(os.path.join(TABLES_DIR, "sensitivity_lognormal.csv"))
    predictions = pd.read_csv(os.path.join(TABLES_DIR, "test_predictions.csv"))
    gbm_sigma = float(np.std(np.log(predictions["GBM"] / predictions["actual"])))

    y_max = max(uniform["avg_waiting_time"].max(), lognormal["avg_waiting_time"].max(), rr_hours * SECONDS_PER_HOUR)
    y_max = y_max / SECONDS_PER_HOUR * 1.18

    panels = [
        (uniform, "(a) Bounded uniform: true × (1 ± ε)", "ε (fraction of true runtime)", None),
        (lognormal, "(b) Log-normal: true × exp(σ·N(0,1))", "σ (log-space error spread)", gbm_sigma),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, (sweep, title, xlabel, fitted_sigma) in zip(axes, panels):
        hours = sweep["avg_waiting_time"] / SECONDS_PER_HOUR
        level = sweep["level"]
        ax.plot(level, hours, color=INK, linewidth=1.6, marker="o", markersize=4.5, label="SRTF with noisy predictions")
        ax.axhline(rr_hours, color=MUTED, linestyle="--", linewidth=1.2, label="Round Robin (q = 4)")
        ax.set_ylim(0, y_max)

        crossed = sweep.loc[hours > rr_hours]
        if crossed.empty:
            ax.annotate("no crossover in range", (level.max(), rr_hours), xytext=(0, -9),
                        textcoords="offset points", ha="right", va="top", fontsize=7.5, color=MUTED)
        else:
            cross = crossed["level"].iloc[0]
            ax.axvline(cross, color=MUTED, linestyle=":", linewidth=1.0)
            ax.annotate(f"crossover ≈ {cross:g}", (cross, y_max), xytext=(4, -4),
                        textcoords="offset points", ha="left", va="top", fontsize=7.5, color=MUTED)

        if fitted_sigma is not None:
            ax.axvline(fitted_sigma, color=INK, linestyle="-.", linewidth=1.0, label=f"fitted GBM, σ ≈ {fitted_sigma:.1f}")
            ax.legend(loc="upper left")

        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")

    axes[0].set_ylabel("Average waiting time (hours)")
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_sensitivity.png")


def figure_predicted_vs_actual():
    predictions = pd.read_csv(os.path.join(TABLES_DIR, "test_predictions.csv"))
    actual = predictions["actual"].to_numpy(dtype=float)
    predicted = predictions["GBM"].to_numpy(dtype=float)
    lo = max(min(actual.min(), predicted.min()), 1.0)
    hi = max(actual.max(), predicted.max()) * 1.5

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(actual, predicted, s=9, color=HIGHLIGHT_FILL, alpha=0.35, linewidths=0, label=f"tasks (n = {len(actual):,})")
    ax.plot([lo, hi], [lo, hi], color=INK, linestyle="--", linewidth=1.2, label="y = x (perfect prediction)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, axis="both")
    ax.set_xlabel("Actual runtime (s)")
    ax.set_ylabel("GBM predicted runtime (s)")
    ax.legend(loc="upper left")
    ax.set_title("GBM predicted vs actual runtime (test set, log-log)", loc="left")
    _save(fig, "fig3_gbm_predicted_vs_actual.png")


def main():
    figure_waiting_time_by_scheduler()
    figure_sensitivity()
    figure_predicted_vs_actual()


if __name__ == "__main__":
    main()
