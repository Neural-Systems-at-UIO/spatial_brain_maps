"""Plot interpolation reliability using atlas-region mean expression."""

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
plot_script = HERE / "2_graph_interpolation_reliability.py"
spec = importlib.util.spec_from_file_location(
    "interpolation_reliability_plot", plot_script
)
plotting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotting)


if __name__ == "__main__":
    plotting.main(
        correlation_column="pearson_r_atlas_regions",
        output_stem=HERE / "interpolation_reliability_atlas_regions",
        ylabel="Pearson correlation across atlas-region means",
        summary_label="Atlas-region",
    )
