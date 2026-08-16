"""Compare reliability in genuinely sampled and interpolated-only voxels."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "interpolation_correlations.csv"
OUTPUT_STEM = HERE / "interpolation_reliability_real_vs_interpolated"

CORRELATION_COLUMNS = {
    "Real-data overlap": "pearson_r_real_data",
    "Interpolated only": "pearson_r_interpolated_only",
}
COLOURS = {
    "Real-data overlap": "#1772b6",
    "Interpolated only": "#d55e00",
}


def load_results(path: Path) -> pd.DataFrame:
    """Load complete rows and reshape the two voxel classes for plotting."""
    results = pd.read_csv(path, on_bad_lines="skip")
    required = {"gene", "comparison_experiment_count", *CORRELATION_COLUMNS.values()}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(
            "Missing columns produced by 1_calculate_interpolation.py: "
            + ", ".join(sorted(missing))
        )

    results = results.dropna(subset=list(required)).copy()
    results["comparison_experiment_count"] = pd.to_numeric(
        results["comparison_experiment_count"], errors="coerce"
    )
    long = results.melt(
        id_vars=["gene", "comparison_experiment_count"],
        value_vars=list(CORRELATION_COLUMNS.values()),
        var_name="correlation_type",
        value_name="pearson_r",
    )
    labels = {column: label for label, column in CORRELATION_COLUMNS.items()}
    long["voxel_class"] = long["correlation_type"].map(labels)
    long["pearson_r"] = pd.to_numeric(long["pearson_r"], errors="coerce")
    return long.dropna(subset=["comparison_experiment_count", "pearson_r"])


def make_plot(results: pd.DataFrame) -> plt.Figure:
    """Plot equal-weighted gene means for both voxel classes."""
    gene_means = results.groupby(
        ["comparison_experiment_count", "gene", "voxel_class"], as_index=False
    )["pearson_r"].mean()
    overall = gene_means.groupby(
        ["comparison_experiment_count", "voxel_class"], as_index=False
    )["pearson_r"].mean()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for voxel_class, values in overall.groupby("voxel_class", sort=False):
        values = values.sort_values("comparison_experiment_count")
        ax.plot(
            values["comparison_experiment_count"],
            values["pearson_r"],
            marker="o",
            linewidth=2.5,
            markersize=5,
            color=COLOURS[voxel_class],
            label=voxel_class,
        )

    sample_counts = sorted(overall["comparison_experiment_count"].unique())
    ax.set_xticks(sample_counts)
    ax.set_xlabel("Number of experiments in comparison average")
    ax.set_ylabel("Pearson correlation with 15-experiment average")
    ax.set_ylim(top=1.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
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
