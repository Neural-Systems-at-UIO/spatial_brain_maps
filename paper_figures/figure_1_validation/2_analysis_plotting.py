import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests  # This is the correct import
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches


def load_and_prepare_data():
    df = pd.read_csv("datafiles/consolidated_registration_results.csv")

    # Melt the dataframe to convert to long format
    raters = [
        "Expert 1",
        "Expert 2",
        "Expert 3",
        "Novice 1",
        "Novice 2",
        "Novice 3",
        "Our Pipeline",
        "ABA",
    ]
    for rater in raters:
        df[f"{rater}_error"] *= 25

    # Melt error columns
    error_df = df.melt(
        id_vars=["brain_id", "section_nr"],
        value_vars=[f"{r}_error" for r in raters],
        var_name="rater",
        value_name="error",
    )
    error_df["rater"] = error_df["rater"].str.replace("_error", "")
    # Melt angle columns
    ml_df = df.melt(
        id_vars=["brain_id", "section_nr"],
        value_vars=[f"{r}_ml" for r in raters],
        var_name="rater",
        value_name="ml_angle",
    )
    ml_df["rater"] = ml_df["rater"].str.replace("_ml", "")

    dv_df = df.melt(
        id_vars=["brain_id", "section_nr"],
        value_vars=[f"{r}_dv" for r in raters],
        var_name="rater",
        value_name="dv_angle",
    )
    dv_df["rater"] = dv_df["rater"].str.replace("_dv", "")

    # Merge all data
    merged_df = error_df.merge(ml_df, on=["brain_id", "section_nr", "rater"])
    merged_df = merged_df.merge(dv_df, on=["brain_id", "section_nr", "rater"])

    # Add human summary stats
    human_stats = df[
        [
            "brain_id",
            "section_nr",
            "human_ml_avg",
            "human_ml_std",
            "human_dv_avg",
            "human_dv_std",
        ]
    ]
    merged_df = merged_df.merge(human_stats, on=["brain_id", "section_nr"])

    # Classify raters into groups
    merged_df["group"] = "Method"
    merged_df.loc[merged_df["rater"].str.startswith("Expert"), "group"] = "Expert"
    merged_df.loc[merged_df["rater"].str.startswith("Novice"), "group"] = "Novice"
    merged_df.loc[merged_df["rater"] == "Our Pipeline", "group"] = "Our Pipeline"
    merged_df.loc[merged_df["rater"] == "ABA", "group"] = "ABA"

    return merged_df


def plot_angle_comparisons(df):
    # Calculate mean angles per brain and rater
    angle_data = (
        df.groupby(["brain_id", "rater", "group"])
        .agg(
            {
                "ml_angle": "mean",
                "dv_angle": "mean",
                "human_ml_avg": "first",
                "human_ml_std": "first",
                "human_dv_avg": "first",
                "human_dv_std": "first",
            }
        )
        .reset_index()
    )

    # Filter for methods we want to compare
    methods = angle_data[angle_data["group"].isin(["Our Pipeline", "ABA"])]

    # Extract data for R squared calculation
    pipeline_data = methods[methods["group"] == "Our Pipeline"]
    aba_data = methods[methods["group"] == "ABA"]

    # Calculate and print R squared for ML and DV angles
    from sklearn.metrics import r2_score

    r2_pipeline_ml = r2_score(pipeline_data["human_ml_avg"], pipeline_data["ml_angle"])
    r2_pipeline_dv = r2_score(pipeline_data["human_dv_avg"], pipeline_data["dv_angle"])
    r2_aba_ml = r2_score(aba_data["human_ml_avg"], aba_data["ml_angle"])
    r2_aba_dv = r2_score(aba_data["human_dv_avg"], aba_data["dv_angle"])
    print(f"R squared (Our Pipeline ML): {r2_pipeline_ml:.3f}")
    print(f"R squared (Our Pipeline DV): {r2_pipeline_dv:.3f}")
    print(f"R squared (ABA ML): {r2_aba_ml:.3f}")
    print(f"R squared (ABA DV): {r2_aba_dv:.3f}")

    # Calculate combined R² for ML and DV together
    # For Our Pipeline
    y_true_pipeline = np.concatenate(
        [pipeline_data["human_ml_avg"], pipeline_data["human_dv_avg"]]
    )
    y_pred_pipeline = np.concatenate(
        [pipeline_data["ml_angle"], pipeline_data["dv_angle"]]
    )
    r2_pipeline_combined = r2_score(y_true_pipeline, y_pred_pipeline)
    print(f"Combined R squared (Our Pipeline ML+DV): {r2_pipeline_combined:.3f}")

    # For ABA
    y_true_aba = np.concatenate([aba_data["human_ml_avg"], aba_data["human_dv_avg"]])
    y_pred_aba = np.concatenate([aba_data["ml_angle"], aba_data["dv_angle"]])
    r2_aba_combined = r2_score(y_true_aba, y_pred_aba)
    print(f"Combined R squared (ABA ML+DV): {r2_aba_combined:.3f}")

    # Create figure with 2 rows (ML and DV) and 2 columns (Our Pipeline and ABA)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ML Angle plots
    # Our Pipeline ML Angle
    pipeline_data = methods[methods["group"] == "Our Pipeline"]
    axes[0, 0].errorbar(
        pipeline_data["human_ml_avg"],
        pipeline_data["ml_angle"],
        xerr=pipeline_data["human_ml_std"],
        fmt="o",
        alpha=0.7,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="gray",
        capsize=3,
    )
    min_val = min(pipeline_data["human_ml_avg"].min(), pipeline_data["ml_angle"].min())
    max_val = max(pipeline_data["human_ml_avg"].max(), pipeline_data["ml_angle"].max())
    axes[0, 0].plot([min_val - 1, max_val + 1], [min_val - 1, max_val + 1], "r--")
    axes[0, 0].set_xlabel("Average Human ML Angle (degrees)")
    axes[0, 0].set_ylabel("Our Pipeline ML Angle (degrees)")
    axes[0, 0].set_title("Our Pipeline ML Angle vs Human Average")

    # ABA ML Angle
    aba_data = methods[methods["group"] == "ABA"]
    axes[0, 1].errorbar(
        aba_data["human_ml_avg"],
        aba_data["ml_angle"],
        xerr=aba_data["human_ml_std"],
        fmt="o",
        alpha=0.7,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="gray",
        capsize=3,
    )
    min_val = min(aba_data["human_ml_avg"].min(), aba_data["ml_angle"].min())
    max_val = max(aba_data["human_ml_avg"].max(), aba_data["ml_angle"].max())
    axes[0, 1].plot([min_val - 1, max_val + 1], [min_val - 1, max_val + 1], "r--")
    axes[0, 1].set_xlabel("Average Human ML Angle (degrees)")
    axes[0, 1].set_ylabel("ABA ML Angle (degrees)")
    axes[0, 1].set_title("ABA ML Angle vs Human Average")

    # DV Angle plots
    # Our Pipeline DV Angle
    axes[1, 0].errorbar(
        pipeline_data["human_dv_avg"],
        pipeline_data["dv_angle"],
        xerr=pipeline_data["human_dv_std"],
        fmt="o",
        alpha=0.7,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="gray",
        capsize=3,
    )
    min_val = min(pipeline_data["human_dv_avg"].min(), pipeline_data["dv_angle"].min())
    max_val = max(pipeline_data["human_dv_avg"].max(), pipeline_data["dv_angle"].max())
    axes[1, 0].plot([min_val - 1, max_val + 1], [min_val - 1, max_val + 1], "r--")
    axes[1, 0].set_xlabel("Average Human DV Angle (degrees)")
    axes[1, 0].set_ylabel("Our Pipeline DV Angle (degrees)")
    axes[1, 0].set_title("Our Pipeline DV Angle vs Human Average")

    # ABA DV Angle
    axes[1, 1].errorbar(
        aba_data["human_dv_avg"],
        aba_data["dv_angle"],
        xerr=aba_data["human_dv_std"],
        fmt="o",
        alpha=0.7,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="gray",
        capsize=3,
    )
    min_val = min(aba_data["human_dv_avg"].min(), aba_data["dv_angle"].min())
    max_val = max(aba_data["human_dv_avg"].max(), aba_data["dv_angle"].max())
    axes[1, 1].plot([min_val - 1, max_val + 1], [min_val - 1, max_val + 1], "r--")
    axes[1, 1].set_xlabel("Average Human DV Angle (degrees)")
    axes[1, 1].set_ylabel("ABA DV Angle (degrees)")
    axes[1, 1].set_title("ABA DV Angle vs Human Average")

    plt.tight_layout()
    plt.savefig("plots/angle_comparisons.png")
    plt.savefig("plots/angle_comparisons.pdf")


def plot_combined_angle_comparisons(df):
    """
    Plot both ML and DV angles in a single scatter plot for each method (Our Pipeline and ABA),
    comparing to human averages, with error bars.
    """
    angle_data = (
        df.groupby(["brain_id", "rater", "group"])
        .agg(
            {
                "ml_angle": "mean",
                "dv_angle": "mean",
                "human_ml_avg": "first",
                "human_ml_std": "first",
                "human_dv_avg": "first",
                "human_dv_std": "first",
            }
        )
        .reset_index()
    )
    methods = angle_data[angle_data["group"].isin(["Our Pipeline", "ABA"])]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    method_names = ["Our Pipeline", "ABA"]
    colors = {"ML": "#1f77b4", "DV": "#ff7f0e"}

    for idx, method in enumerate(method_names):
        method_data = methods[methods["group"] == method]
        # ML
        axes[idx].errorbar(
            method_data["human_ml_avg"],
            method_data["ml_angle"],
            xerr=method_data["human_ml_std"],
            fmt="o",
            color=colors["ML"],
            alpha=0.7,
            markerfacecolor="white",
            markeredgecolor=colors["ML"],
            ecolor=colors["ML"],
            capsize=3,
            label="ML",
        )
        # DV
        axes[idx].errorbar(
            method_data["human_dv_avg"],
            method_data["dv_angle"],
            xerr=method_data["human_dv_std"],
            fmt="s",
            color=colors["DV"],
            alpha=0.7,
            markerfacecolor="white",
            markeredgecolor=colors["DV"],
            ecolor=colors["DV"],
            capsize=3,
            label="DV",
        )
        # 1:1 lines for both
        min_val = min(
            method_data["human_ml_avg"].min(),
            method_data["ml_angle"].min(),
            method_data["human_dv_avg"].min(),
            method_data["dv_angle"].min(),
        )
        max_val = max(
            method_data["human_ml_avg"].max(),
            method_data["ml_angle"].max(),
            method_data["human_dv_avg"].max(),
            method_data["dv_angle"].max(),
        )
        axes[idx].plot(
            [min_val - 1, max_val + 1], [min_val - 1, max_val + 1], "k--", alpha=0.5
        )
        axes[idx].set_xlabel("Average Human Angle (degrees)")
        axes[idx].set_ylabel(f"{method} Angle (degrees)")
        axes[idx].set_title(f"{method}: ML and DV Angles vs Human Average")
        axes[idx].legend(
            handles=[
                mpatches.Patch(color=colors["ML"], label="ML"),
                mpatches.Patch(color=colors["DV"], label="DV"),
            ],
            loc="upper left",
        )
        axes[idx].grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.savefig("plots/combined_angle_comparisons.png")
    plt.savefig("plots/combined_angle_comparisons.pdf")


def plot_error_distributions(df):
    # Create a single figure without broken y-axis
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500))

    # Filter data for plotting
    plot_data = df[df["group"].isin(["Expert", "Novice", "Our Pipeline", "ABA"])]

    # Create swarm plot with all data
    sns.swarmplot(
        x="rater", y="error", hue="group", data=plot_data, size=3, legend=False, ax=ax
    )

    # Calculate group means and positions
    group_means = plot_data.groupby("group")["error"].mean()
    group_positions = {
        "Expert": [0, 1, 2],  # Positions of Expert raters on x-axis
        "Novice": [3, 4, 5],  # Positions of Novice raters
        "Our Pipeline": [6],  # Position of Our Pipeline
        "ABA": [7],  # Position of ABA
    }

    # Plot horizontal bars for each group mean
    for group, positions in group_positions.items():
        x_min = min(positions) - 0.5
        x_max = max(positions) + 0.5
        mean_val = group_means[group]
        ax.hlines(
            mean_val, x_min, x_max, colors="red", linestyles="-", linewidth=2, alpha=0.7
        )

    # Extract error values for each group for statistical tests
    expert_errors = plot_data[plot_data["group"] == "Expert"]["error"]
    novice_errors = plot_data[plot_data["group"] == "Novice"]["error"]
    pipeline_errors = plot_data[plot_data["rater"] == "Our Pipeline"]["error"]
    aba_errors = plot_data[plot_data["rater"] == "ABA"]["error"]

    # Prepare data for ANOVA and Tukey's HSD
    anova_data = pd.DataFrame(
        {
            "error": pd.concat(
                [expert_errors, novice_errors, pipeline_errors, aba_errors]
            ),
            "group": ["Expert"] * len(expert_errors)
            + ["Novice"] * len(novice_errors)
            + ["Our Pipeline"] * len(pipeline_errors)
            + ["ABA"] * len(aba_errors),
        }
    )

    # Define comparisons and their positions - using improved bracket formatting
    comparisons = [
        ("Expert vs ABA", "Expert", "ABA", (0, 2), 7),  # Expert group span to ABA
        (
            "Our Pipeline vs Expert",
            "Our Pipeline",
            "Expert",
            6,
            (0, 2),
        ),  # Pipeline to Expert group span
        ("Novice vs ABA", "Novice", "ABA", (3, 5), 7),  # Novice group span to ABA
        (
            "Our Pipeline vs Novice",
            "Our Pipeline",
            "Novice",
            6,
            (3, 5),
        ),  # Pipeline to Novice group span
        (
            "Our Pipeline vs ABA",
            "Our Pipeline",
            "ABA",
            6,
            7,
        ),  # Pipeline to ABA (both single points)
    ]

    # Calculate significance bars using Tukey's HSD
    y_max = plot_data["error"].max()
    bar_height = y_max * 0.06  # Slightly larger for better spacing

    # Perform Tukey's HSD test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    tukey_results = pairwise_tukeyhsd(anova_data["error"], anova_data["group"])

    # Extract results from Tukey's test
    tukey_data = pd.DataFrame(
        data=tukey_results._results_table.data[1:],
        columns=tukey_results._results_table.data[0],
    )
    prev_groups = []
    for i, (name, group1, group2, pos1, pos2) in enumerate(comparisons):
        # Find the Tukey result for this pair
        tukey_row = tukey_data[
            (tukey_data["group1"] == group1) & (tukey_data["group2"] == group2)
        ]
        if len(tukey_row) == 0:
            tukey_row = tukey_data[
                (tukey_data["group1"] == group2) & (tukey_data["group2"] == group1)
            ]

        if len(tukey_row) > 0:
            p_val = float(tukey_row["p-adj"].values[0])
            reject = tukey_row["reject"].values[0]

            # Determine significance level using Tukey's adjusted p-values
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            y_pos = y_max + (i + 1) * bar_height * 2.2

            # Calculate bar position
            if group2 in prev_groups:
                y_pos -= 5 * 25
            if (group1 == "Our Pipeline") & (group2 == "ABA"):
                y_pos -= 10 * 25

            # Handle positions - if it's a tuple, it's a group span; if not, it's a single point
            if isinstance(pos1, tuple):
                x1_start, x1_end = pos1
                x1_mid = (x1_start + x1_end) / 2
            else:
                x1_start = x1_end = x1_mid = pos1

            if isinstance(pos2, tuple):
                x2_start, x2_end = pos2
                x2_mid = (x2_start + x2_end) / 2
            else:
                x2_start = x2_end = x2_mid = pos2

            # Draw main horizontal connecting line
            ax.plot([x1_mid, x2_mid], [y_pos, y_pos], "k-", linewidth=1.2)

            # Draw improved brackets
            bracket_height = bar_height * 0.4
            stem_height = (
                bar_height * 0.4
            )  # Small vertical stems connecting to brackets

            if isinstance(pos1, tuple):
                # Left bracket - U-shape below the line, connected by stem
                bracket_y = y_pos - stem_height - (4 * 25)
                # Vertical stem connecting main line to bracket
                if group1 not in prev_groups:
                    ax.plot(
                        [x1_mid, x1_mid],
                        [y_pos, y_pos - stem_height],
                        "k-",
                        linewidth=1.2,
                    )
                    # U-shaped bracket below
                    # Get color for the group from the swarmplot palette
                    palette = sns.color_palette()
                    group_color_map = {
                        "Expert": palette[0],
                        "Novice": palette[1],
                        "Our Pipeline": palette[2],
                        "ABA": palette[3],
                    }
                    colour = group_color_map.get(group1, "k")
                    ax.plot(
                        [x1_start, x1_end],
                        [bracket_y, bracket_y],
                        color=colour,
                        linewidth=3,
                    )
                    # ax.plot([x1_start, x1_start], [bracket_y, bracket_y - bracket_height], 'k-', linewidth=1.2)
                    # ax.plot([x1_end, x1_end], [bracket_y, bracket_y - bracket_height], 'k-', linewidth=11)
                    prev_groups.append(group1)
                else:
                    ax.plot(
                        [x1_mid, x1_mid],
                        [y_pos, y_pos - (stem_height * 6)],
                        "k-",
                        linewidth=1.2,
                    )

            else:
                ax.plot(
                    [x1_mid, x1_mid],
                    [y_pos - bracket_height / 2, y_pos + bracket_height / 2],
                    "k-",
                    linewidth=1.2,
                )

            if isinstance(pos2, tuple):
                # Right bracket - U-shape below the line, connected by stem
                bracket_y = y_pos - stem_height
                # Vertical stem connecting main line to bracket
                if group2 not in prev_groups:
                    ax.plot(
                        [x2_mid, x2_mid],
                        [y_pos, y_pos - stem_height],
                        "k-",
                        linewidth=1.2,
                    )

                    # U-shaped bracket below
                    # ax.plot([x2_start, x2_start], [bracket_y, bracket_y - bracket_height], 'k-', linewidth=1.2)
                    # ax.plot([x2_start, x2_end], [bracket_y, bracket_y], 'k-', linewidth=1.2)
                    # ax.plot([x2_end, x2_end], [bracket_y, bracket_y - bracket_height], 'k-', linewidth=1.2)
                    prev_groups.append(group2)
                else:
                    ax.plot(
                        [x2_mid, x2_mid],
                        [y_pos, y_pos - (stem_height * 6)],
                        "k-",
                        linewidth=1.2,
                    )

            else:
                ax.plot(
                    [x2_mid, x2_mid],
                    [y_pos - bracket_height / 2, y_pos + bracket_height / 2],
                    "k-",
                    linewidth=1.2,
                )

            # Add significance text and p-value above the connecting line
            overall_mid = (x1_mid + x2_mid) / 2
            if p_val < 0.001:
                p_text = "p<0.001"
            else:
                p_text = f"p={p_val:.3f}"
            sig_with_p = f"{sig_text} ({p_text})"
            ax.text(
                overall_mid,
                y_pos + bar_height * 0.3,
                sig_with_p,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Adjust y-axis limits to accommodate significance bars
    ax.set_ylim(ax.get_ylim()[0], y_max + len(comparisons) * bar_height * 2.5)

    # Add a subtle background grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.2, zorder=-1)

    # Add a subtle shading to the background
    ax.set_facecolor("#f9f9f9")

    # Set labels and title
    ax.set_ylabel("Registration Error (microns)")
    ax.set_xlabel("Rater")
    fig.suptitle("Registration Error Distribution by Rater", fontsize=16)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    # Adjust the layout to ensure proper spacing for title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("plots/error_distributions.png", dpi=300, bbox_inches="tight")
    plt.savefig("plots/error_distributions.pdf", dpi=300, bbox_inches="tight")


def perform_statistical_tests(df):
    print("\n=== Statistical Test Results ===")

    # Extract error values for each group
    expert_errors = df[df["group"] == "Expert"]["error"]
    novice_errors = df[df["group"] == "Novice"]["error"]
    pipeline_errors = df[df["rater"] == "Our Pipeline"]["error"]
    aba_errors = df[df["rater"] == "ABA"]["error"]

    # Create a dataframe with group labels for ANOVA
    anova_data = pd.DataFrame(
        {
            "error": pd.concat(
                [expert_errors, novice_errors, pipeline_errors, aba_errors]
            ),
            "group": ["Expert"] * len(expert_errors)
            + ["Novice"] * len(novice_errors)
            + ["Our Pipeline"] * len(pipeline_errors)
            + ["ABA"] * len(aba_errors),
        }
    )
    # Perform one-way ANOVA as the primary test
    groups = [
        anova_data[anova_data["group"] == g]["error"]
        for g in ["Expert", "Novice", "Our Pipeline", "ABA"]
    ]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"One-way ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    # If ANOVA is significant, perform post-hoc Tukey's HSD test
    if p_val < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey_results = pairwise_tukeyhsd(anova_data["error"], anova_data["group"])
        print("\nTukey's HSD Post-hoc Test:")
        print(tukey_results)


def plot_statistical_comparisons(df):
    """
    Create a grid visualization showing statistical comparisons between groups,
    using ANOVA with Tukey's HSD for post-hoc tests.
    Only show comparisons once in a triangular grid.
    """
    # Extract error values for each group
    expert_errors = df[df["group"] == "Expert"]["error"]
    novice_errors = df[df["group"] == "Novice"]["error"]
    pipeline_errors = df[df["rater"] == "Our Pipeline"]["error"]
    aba_errors = df[df["rater"] == "ABA"]["error"]

    # Prepare data for ANOVA and Tukey's HSD
    anova_data = pd.DataFrame(
        {
            "error": pd.concat(
                [expert_errors, novice_errors, pipeline_errors, aba_errors]
            ),
            "group": ["Expert"] * len(expert_errors)
            + ["Novice"] * len(novice_errors)
            + ["Our Pipeline"] * len(pipeline_errors)
            + ["ABA"] * len(aba_errors),
        }
    )

    # Perform ANOVA
    groups = ["Expert", "Novice", "Our Pipeline", "ABA"]
    group_data = {
        "Expert": expert_errors,
        "Novice": novice_errors,
        "Our Pipeline": pipeline_errors,
        "ABA": aba_errors,
    }

    # Calculate group means for the plot
    group_means = {g: group_data[g].mean() for g in groups}

    # Create figure
    fig, axes = plt.subplots(len(groups), len(groups), figsize=(12, 12))

    # Turn off all axes by default
    for i in range(len(groups)):
        for j in range(len(groups)):
            axes[i, j].axis("off")

    # Add group names and mean values on diagonal
    for i, group in enumerate(groups):
        mean_val = group_means[group]
        axes[i, i].text(
            0.5,
            0.5,
            f"{group}\nMean: {mean_val:.1f}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        axes[i, i].axis("on")
        axes[i, i].set_xticks([])
        axes[i, i].set_yticks([])
        axes[i, i].set_facecolor("#e6f2ff")  # Light blue for diagonal

    try:
        # Perform Tukey's HSD test
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        tukey_results = pairwise_tukeyhsd(anova_data["error"], anova_data["group"])

        # Extract results from Tukey's test
        tukey_data = pd.DataFrame(
            data=tukey_results._results_table.data[1:],
            columns=tukey_results._results_table.data[0],
        )

        # Fill in only the upper triangle of the grid (above diagonal) with Tukey's HSD results
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):  # Only upper triangle (i < j)
                group1 = groups[i]
                group2 = groups[j]

                # Find the Tukey result for this pair
                tukey_row = tukey_data[
                    (tukey_data["group1"] == group1) & (tukey_data["group2"] == group2)
                ]
                if len(tukey_row) == 0:
                    tukey_row = tukey_data[
                        (tukey_data["group1"] == group2)
                        & (tukey_data["group2"] == group1)
                    ]

                if len(tukey_row) > 0:
                    p_val = float(tukey_row["p-adj"].values[0])
                    reject = tukey_row["reject"].values[0]

                    # Format p-value with stars for significance
                    if p_val < 0.001:
                        sig_str = "p < 0.001 ***"
                    elif p_val < 0.01:
                        sig_str = f"p = {p_val:.3f} **"
                    elif p_val < 0.05:
                        sig_str = f"p = {p_val:.3f} *"
                    else:
                        sig_str = f"p = {p_val:.3f} (ns)"

                    # Calculate mean difference
                    diff = group_means[group1] - group_means[group2]

                    # Create the comparison text
                    comparison_text = (
                        f"{group1} vs {group2}\nMean diff: {diff:.1f}\n{sig_str}"
                    )

                    # Add to plot - only in upper triangle
                    axes[i, j].axis("on")
                    axes[i, j].text(
                        0.5, 0.5, comparison_text, ha="center", va="center", fontsize=10
                    )
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])

                    # Color based on significance
                    if reject:
                        axes[i, j].set_facecolor("#ffeeee")  # Light red for significant
                    else:
                        axes[i, j].set_facecolor(
                            "#eeffee"
                        )  # Light green for non-significant

        plt.suptitle(
            "Statistical Comparisons of Registration Error Between Groups\n(ANOVA with Tukey's HSD)",
            fontsize=16,
        )

    except ImportError:
        # Fallback to t-tests if statsmodels is not available
        print("Note: statsmodels not available, using t-tests instead of Tukey's HSD")

        # Compute statistical tests for each pair - only in upper triangle
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):  # Only upper triangle (i < j)
                group1 = groups[i]
                group2 = groups[j]

                # Perform t-test
                t_stat, p_val = stats.ttest_ind(
                    group_data[group1], group_data[group2], equal_var=False
                )

                # Format p-value with stars for significance
                if p_val < 0.001:
                    sig_str = "p < 0.001 ***"
                elif p_val < 0.01:
                    sig_str = f"p = {p_val:.3f} **"
                elif p_val < 0.05:
                    sig_str = f"p = {p_val:.3f} *"
                else:
                    sig_str = f"p = {p_val:.3f} (ns)"

                # Calculate mean difference
                diff = group_means[group1] - group_means[group2]

                # Create the comparison text
                comparison_text = (
                    f"{group1} vs {group2}\nMean diff: {diff:.1f}\n{sig_str}"
                )

                # Add to plot - only in upper triangle
                axes[i, j].axis("on")
                axes[i, j].text(
                    0.5, 0.5, comparison_text, ha="center", va="center", fontsize=10
                )
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

                # Color based on significance
                if p_val < 0.05:
                    axes[i, j].set_facecolor("#ffeeee")  # Light red for significant
                else:
                    axes[i, j].set_facecolor(
                        "#eeffee"
                    )  # Light green for non-significant

        plt.suptitle(
            "Statistical Comparisons of Registration Error Between Groups\n(Pairwise t-tests)",
            fontsize=16,
        )

    # Removed the side labels

    plt.tight_layout()
    plt.savefig("plots/statistical_comparisons.png", dpi=300, bbox_inches="tight")
    plt.savefig("plots/statistical_comparisons.pdf", dpi=300, bbox_inches="tight")


# Main analysis
df = load_and_prepare_data()

# Generate plots
plot_angle_comparisons(df)
plot_combined_angle_comparisons(df)
plot_error_distributions(df)
plot_statistical_comparisons(df)

# Perform statistical tests
perform_statistical_tests(df)

# Print summary statistics
print("\n=== Summary Statistics ===")
print(df.groupby(["group", "rater"])["error"].agg(["mean", "median", "std"]))
