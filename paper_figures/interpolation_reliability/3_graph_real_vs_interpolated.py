"""Compare reliability in genuinely sampled and interpolated-only voxels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
        value_name="masked_pearson_r",
    )
    labels = {column: label for label, column in CORRELATION_COLUMNS.items()}
    long["voxel_class"] = long["correlation_type"].map(labels)
    long["masked_pearson_r"] = pd.to_numeric(long["masked_pearson_r"], errors="coerce")
    return long.dropna(subset=["comparison_experiment_count", "masked_pearson_r"])


def make_plot(results: pd.DataFrame) -> plt.Figure:
    """Compare gene-level means for one experiment using two bars."""
    one_experiment = results.loc[results["comparison_experiment_count"] == 1]
    if one_experiment.empty:
        raise ValueError("No results found for comparison_experiment_count = 1")

    gene_means = one_experiment.groupby(["gene", "voxel_class"], as_index=False)[
        "masked_pearson_r"
    ].mean()
    order = list(CORRELATION_COLUMNS)
    summary = (
        gene_means.groupby("voxel_class")["masked_pearson_r"]
        .agg(["mean", "std"])
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    x_positions = np.arange(len(order))
    ax.bar(
        x_positions,
        summary["mean"],
        yerr=summary["std"],
        width=0.62,
        capsize=5,
        color=[COLOURS[label] for label in order],
        edgecolor="#333333",
        linewidth=0.8,
        error_kw={"elinewidth": 1.4},
    )
    for x_position, voxel_class in zip(x_positions, order):
        values = gene_means.loc[
            gene_means["voxel_class"] == voxel_class, "masked_pearson_r"
        ].to_numpy()
        offsets = np.linspace(-0.11, 0.11, len(values))
        ax.scatter(
            x_position + offsets,
            values,
            s=20,
            color="#202020",
            alpha=0.65,
            zorder=3,
        )

    ax.set_xticks(x_positions, order)
    ax.set_ylabel("Pearson correlation with 15-experiment average")
    ax.set_ylim(top=1.0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
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
        f"Plotted the one-experiment comparison for {results['gene'].nunique()} genes."
    )


if __name__ == "__main__":
    main()
