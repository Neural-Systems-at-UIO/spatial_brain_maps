"""Plot interpolation reliability as the number of experiments increases."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "interpolation_correlations.csv"
OUTPUT_STEM = HERE / "interpolation_reliability"


def load_results(path: Path) -> pd.DataFrame:
    """Load complete result rows, ignoring a final row still being written."""
    results = pd.read_csv(path, on_bad_lines="skip")
    required = ["gene", "comparison_experiment_count", "pearson_r"]
    missing = set(required).difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    results = results.dropna(subset=required).copy()
    results["comparison_experiment_count"] = pd.to_numeric(
        results["comparison_experiment_count"], errors="coerce"
    )
    results["pearson_r"] = pd.to_numeric(results["pearson_r"], errors="coerce")
    return results.dropna(subset=["comparison_experiment_count", "pearson_r"])


def make_plot(results: pd.DataFrame) -> plt.Figure:
    # Average the random experiment combinations within each gene and sample count.
    gene_means = (
        results.groupby(["comparison_experiment_count", "gene"], as_index=False)[
            "pearson_r"
        ]
        .mean()
        .sort_values(["gene", "comparison_experiment_count"])
    )

    # Average gene means so every gene has equal influence, including while the CSV
    # is only partly populated.
    overall = gene_means.groupby("comparison_experiment_count", as_index=False)[
        "pearson_r"
    ].mean()

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    genes = list(gene_means.groupby("gene"))
    colour_map = plt.get_cmap("turbo")
    colours = []
    for i in range(len(genes)):
        red, green, blue, alpha = colour_map(i / max(len(genes) - 1, 1))
        # Darken the palette so pale cyan/yellow lines remain legible on white.
        colours.append((red * 0.5, green * 0.5, blue * 0.5, alpha))
    end_labels = []
    for colour, (gene, values) in zip(colours, genes):
        ax.plot(
            values["comparison_experiment_count"],
            values["pearson_r"],
            linewidth=1.5,
            alpha=0.58,
            color=colour,
        )
        last = values.iloc[-1]
        end_labels.append(
            (
                str(gene),
                float(last["comparison_experiment_count"]),
                float(last["pearson_r"]),
                colour,
                0.8,
            )
        )

    ax.plot(
        overall["comparison_experiment_count"],
        overall["pearson_r"],
        linewidth=3.2,
        color="#202020",
        zorder=10,
    )
    end_labels.append(
        (
            "Overall mean",
            float(overall.iloc[-1]["comparison_experiment_count"]),
            float(overall.iloc[-1]["pearson_r"]),
            "#202020",
            1.0,
        )
    )

    sample_counts = sorted(gene_means["comparison_experiment_count"].unique())
    ax.set_xticks(sample_counts)
    ax.set_xlabel("Number of experiments in comparison average")
    ax.set_ylabel("Pearson correlation with 15-experiment average")
    ax.set_ylim(top=1.0)
    x_end = max(sample_counts)
    ax.set_xlim(min(sample_counts) - 0.4, x_end + 1.65)

    # Spread nearby endpoint labels vertically while retaining a short connector
    # to the true endpoint. This remains readable as more genes are added.
    y_low, y_high = ax.get_ylim()
    minimum_gap = 0.028 * (y_high - y_low)
    positioned = []
    for label in sorted(end_labels, key=lambda item: item[2]):
        label_y = label[2]
        if positioned:
            label_y = max(label_y, positioned[-1][1] + minimum_gap)
        positioned.append((label, label_y))
    overflow = positioned[-1][1] - (y_high - minimum_gap / 2)
    if overflow > 0:
        positioned = [(label, y - overflow) for label, y in positioned]

    for (label, actual_x, actual_y, colour, alpha), label_y in positioned:
        ax.plot(
            [actual_x, x_end + 0.15],
            [actual_y, label_y],
            color=colour,
            alpha=alpha,
            linewidth=1,
            clip_on=False,
        )
        ax.text(
            x_end + 0.2,
            label_y,
            label,
            color=colour,
            alpha=alpha,
            va="center",
            fontsize=9,
            fontweight="bold" if label == "Overall mean" else "normal",
        )

    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def main() -> None:
    results = load_results(INPUT_CSV)
    if results.empty:
        raise ValueError(f"No complete result rows found in {INPUT_CSV}")

    figure = make_plot(results)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(
        f"Plotted {results['gene'].nunique()} genes and "
        f"{results['comparison_experiment_count'].nunique()} sample counts."
    )


if __name__ == "__main__":
    main()
