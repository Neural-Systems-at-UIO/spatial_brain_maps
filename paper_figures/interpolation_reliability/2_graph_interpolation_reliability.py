"""Plot interpolation reliability as the number of experiments increases."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "interpolation_correlations.csv"
OUTPUT_STEM = HERE / "interpolation_reliability"


def load_results(
    path: Path, correlation_column: str = "pearson_r_brain"
) -> pd.DataFrame:
    """Load complete result rows, ignoring a final row still being written."""
    results = pd.read_csv(path, on_bad_lines="skip")
    required = ["gene", "comparison_experiment_count", correlation_column]
    missing = set(required).difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    results = results.dropna(subset=required).copy()
    results["comparison_experiment_count"] = pd.to_numeric(
        results["comparison_experiment_count"], errors="coerce"
    )
    results[correlation_column] = pd.to_numeric(
        results[correlation_column], errors="coerce"
    )
    results = results.dropna(subset=["comparison_experiment_count", correlation_column])
    results["pearson_r"] = results.pop(correlation_column)
    return results


def make_plot(
    results: pd.DataFrame,
    ylabel: str = "Pearson correlation within brain voxels",
) -> plt.Figure:
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
    ax.set_ylabel(ylabel)
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


def main(
    correlation_column: str = "pearson_r_brain",
    output_stem: Path = OUTPUT_STEM,
    ylabel: str = "Pearson correlation within brain voxels",
    summary_label: str = "Brain-masked",
) -> None:
    results = load_results(INPUT_CSV, correlation_column)
    if results.empty:
        raise ValueError(f"No complete result rows found in {INPUT_CSV}")

    gene_lines = results.groupby(
        ["gene", "comparison_experiment_count"], as_index=False
    )["pearson_r"].mean()
    statistics = gene_lines.groupby("comparison_experiment_count")["pearson_r"].agg(
        ["min", "max", "mean", "std"]
    )
    statistics = statistics.rename(columns={"std": "gene_mean_std"})
    statistics["experiment_std"] = results.groupby("comparison_experiment_count")[
        "pearson_r"
    ].std()
    groups = gene_lines.groupby("comparison_experiment_count")["pearson_r"]
    min_genes = gene_lines.loc[
        groups.idxmin(), ["comparison_experiment_count", "gene"]
    ].set_index("comparison_experiment_count")["gene"]
    max_genes = gene_lines.loc[
        groups.idxmax(), ["comparison_experiment_count", "gene"]
    ].set_index("comparison_experiment_count")["gene"]
    statistics.insert(1, "min_gene", min_genes)
    statistics.insert(3, "max_gene", max_genes)
    statistics.index.name = "number_of_experiments"
    print(f"{summary_label} Pearson correlation summary across gene means:")
    print(statistics.to_string(float_format=lambda value: f"{value:.4f}"))

    figure = make_plot(results, ylabel)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(
        f"Plotted {results['gene'].nunique()} genes and "
        f"{results['comparison_experiment_count'].nunique()} sample counts."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-regions",
        action="store_true",
        help="plot correlations across atlas-region means instead of brain voxels",
    )
    arguments = parser.parse_args()
    if arguments.atlas_regions:
        main(
            correlation_column="pearson_r_atlas_regions",
            output_stem=HERE / "interpolation_reliability_atlas_regions",
            ylabel="Pearson correlation across atlas-region means",
            summary_label="Atlas-region",
        )
    else:
        main()
