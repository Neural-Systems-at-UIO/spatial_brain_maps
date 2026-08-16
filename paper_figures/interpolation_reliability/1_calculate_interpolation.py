"""Correlation between 1--9 experiment averages and a disjoint 10-experiment average."""

import json
from multiprocessing import Pool
from pathlib import Path

import brainglobe_atlasapi
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.stats import pearsonr

from generate_gene_data import id_to_volume


# Analysis settings
SEED = 20260812
SAMPLES_PER_SIZE = 100
REFERENCE_SIZE = 15
COMPARISON_SIZES = range(1, 11)
RESOLUTION = 25
# Correlate on a regular 100-um grid. Set to 1 to use every 25-um voxel.
CORRELATION_STRIDE = 4
# Volume reconstruction is memory-bound; each worker owns several full brain
# arrays. Running these concurrently can exhaust RAM even though the later
# analysis uses memory-mapped files.
VOLUME_WORKERS = 1

# Data paths
ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "generate_gene_data/metadata/metadata.csv"
IMAGE_FOLDER = Path(
    "/media/harrycarey/Elements/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH"
)
REGISTRATION_FOLDER = Path(
    "/media/harrycarey/Elements/Allen_Realignment_EBRAINS_dataset/registration_data"
)
CACHE_FOLDER = Path(
    "/media/harrycarey/Elements/spatial_brain_maps/"
    "interpolation_reliability/experiment_volumes_25um"
)
OUTPUT_PATH = Path(__file__).with_name("interpolation_correlations.csv")
SUMMARY_PATH = Path(__file__).with_name("interpolation_correlations_summary.csv")


def make_volume(experiment_id):
    """Cache an interpolated volume and a mask of its genuinely sampled voxels."""
    path = CACHE_FOLDER / f"{experiment_id}.npy"
    mask_path = CACHE_FOLDER / f"{experiment_id}.real_data_mask.npy"
    if path.exists() and mask_path.exists():
        return path, mask_path

    # Existing caches predate the real-data masks. Recover fv without repeating
    # the expensive nearest-neighbour interpolation when the volume is present.
    volume_exists = path.exists()
    action = "Recovering real-data mask for" if volume_exists else "Constructing"
    print(f"{action} experiment {experiment_id}", flush=True)
    volume, frequencies = id_to_volume(
        experiment_id,
        str(IMAGE_FOLDER),
        str(REGISTRATION_FOLDER),
        resolution=RESOLUTION,
        mode="expression",
        return_frequencies=True,
        missing_fill=0,
        do_interpolation=not volume_exists,
        k=5,
    )
    temporary_path = CACHE_FOLDER / f"{experiment_id}.tmp.npy"
    temporary_mask_path = CACHE_FOLDER / f"{experiment_id}.real_data_mask.tmp.npy"
    if not volume_exists:
        np.save(temporary_path, volume.astype(np.float32))
        temporary_path.replace(path)
    np.save(temporary_mask_path, frequencies.astype(bool))
    temporary_mask_path.replace(mask_path)
    return path, mask_path


def average_volumes(experiment_ids, volumes):
    sampled = volumes[experiment_ids[0]][
        ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
    ]
    average = np.zeros_like(sampled, dtype=np.float32)
    for experiment_id in experiment_ids:
        average += volumes[experiment_id][
            ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
        ]
    return average / len(experiment_ids)


def union_masks(experiment_ids, masks):
    """Return voxels containing real data in at least one selected experiment."""
    sampled = masks[experiment_ids[0]][
        ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
    ]
    union = np.zeros_like(sampled, dtype=bool)
    for experiment_id in experiment_ids:
        union |= masks[experiment_id][
            ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
        ]
    return union


def masked_correlation(comparison, reference, mask):
    """Return Pearson r and voxel count for a boolean voxel subset."""
    count = int(mask.sum())
    if count < 2:
        return np.nan, count
    return pearsonr(comparison[mask], reference[mask]).statistic, count


def get_atlas_annotation(resolution):
    """Return atlas region IDs in the reconstructed-volume orientation."""
    if resolution < 25:
        atlas = brainglobe_atlasapi.BrainGlobeAtlas(
            "ccfv3augmented_mouse_10um"
        ).annotation
        atlas_resolution = 10
    else:
        atlas = brainglobe_atlasapi.BrainGlobeAtlas(
            "ccfv3augmented_mouse_25um"
        ).annotation
        atlas_resolution = 25
    atlas = np.transpose(atlas, [2, 0, 1])[::-1, ::-1, ::-1]
    scale = atlas_resolution / resolution
    return zoom(atlas, scale, order=0)


def atlas_region_profile(volume, atlas_labels, region_ids):
    """Average voxel expression within each nonzero atlas region."""
    labels = atlas_labels.reshape(-1)
    values = volume.reshape(-1)
    sums = np.bincount(labels, weights=values, minlength=int(labels.max()) + 1)
    counts = np.bincount(labels, minlength=len(sums))
    return (sums[region_ids] / counts[region_ids]).astype(np.float32)


def average_profiles(experiment_ids, profiles):
    """Average precomputed atlas-region profiles across experiments."""
    return np.mean(
        [profiles[experiment_id] for experiment_id in experiment_ids], axis=0
    )


CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv(METADATA_PATH)
atlas_labels = get_atlas_annotation(RESOLUTION)[
    ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
]
brain_mask = atlas_labels != 0
atlas_region_ids = np.unique(atlas_labels[brain_mask])
gene_counts = metadata.groupby("gene")["experiment_id"].nunique()
genes = gene_counts[gene_counts.isin([25, 26])].index
results = []

for gene_index, gene in enumerate(genes):
    rng = np.random.default_rng(SEED + gene_index)
    all_ids = sorted(metadata.loc[metadata["gene"] == gene, "experiment_id"].unique())
    excluded_id = None
    if len(all_ids) == 26:
        excluded_id = int(rng.choice(all_ids))
        all_ids.remove(excluded_id)

    print(f"Processing {gene}; excluded experiment: {excluded_id}", flush=True)
    if VOLUME_WORKERS == 1:
        # Avoid forking after NumPy/SciPy and the atlas have initialized native
        # thread pools. Apart from adding overhead, that can deadlock on a futex.
        cache_paths = [make_volume(experiment_id) for experiment_id in all_ids]
    else:
        with Pool(VOLUME_WORKERS) as pool:
            cache_paths = pool.map(make_volume, all_ids)
    paths = dict(zip(all_ids, (item[0] for item in cache_paths)))
    mask_paths = dict(zip(all_ids, (item[1] for item in cache_paths)))
    volumes = {
        experiment_id: np.load(path, mmap_mode="r")
        for experiment_id, path in paths.items()
    }
    real_data_masks = {
        experiment_id: np.load(path, mmap_mode="r")
        for experiment_id, path in mask_paths.items()
    }
    region_profiles = {
        experiment_id: atlas_region_profile(
            volume[::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE],
            atlas_labels,
            atlas_region_ids,
        )
        for experiment_id, volume in volumes.items()
    }
    voxel_count = next(iter(volumes.values()))[
        ::CORRELATION_STRIDE, ::CORRELATION_STRIDE, ::CORRELATION_STRIDE
    ].size

    for comparison_size in COMPARISON_SIZES:
        pairs = set()
        while len(pairs) < SAMPLES_PER_SIZE:
            reference_ids = tuple(
                sorted(
                    int(i) for i in rng.choice(all_ids, REFERENCE_SIZE, replace=False)
                )
            )
            available_ids = [i for i in all_ids if i not in reference_ids]
            comparison_ids = tuple(
                sorted(
                    int(i)
                    for i in rng.choice(available_ids, comparison_size, replace=False)
                )
            )
            pairs.add((reference_ids, comparison_ids))

        for sample, (reference_ids, comparison_ids) in enumerate(pairs):
            assert set(reference_ids).isdisjoint(comparison_ids)
            reference = average_volumes(reference_ids, volumes)
            comparison = average_volumes(comparison_ids, volumes)
            # A comparison-average voxel is real if any experiment contributing
            # to that average sampled it; all remaining brain voxels are supplied
            # exclusively by interpolation.
            real_data_mask = union_masks(comparison_ids, real_data_masks) & brain_mask
            interpolated_only_mask = brain_mask & ~real_data_mask
            real_r, real_count = masked_correlation(
                comparison, reference, real_data_mask
            )
            interpolated_r, interpolated_count = masked_correlation(
                comparison, reference, interpolated_only_mask
            )
            brain_r, brain_count = masked_correlation(comparison, reference, brain_mask)
            reference_regions = average_profiles(reference_ids, region_profiles)
            comparison_regions = average_profiles(comparison_ids, region_profiles)
            region_r = pearsonr(comparison_regions, reference_regions).statistic
            results.append(
                {
                    "gene": gene,
                    "comparison_experiment_count": comparison_size,
                    "sample": sample,
                    "pearson_r": pearsonr(
                        comparison.reshape(-1), reference.reshape(-1)
                    ).statistic,
                    "voxel_count": voxel_count,
                    "pearson_r_brain": brain_r,
                    "voxel_count_brain": brain_count,
                    "pearson_r_atlas_regions": region_r,
                    "atlas_region_count": len(atlas_region_ids),
                    "pearson_r_real_data": real_r,
                    "voxel_count_real_data": real_count,
                    "pearson_r_interpolated_only": interpolated_r,
                    "voxel_count_interpolated_only": interpolated_count,
                    "reference_experiment_ids": json.dumps(reference_ids),
                    "comparison_experiment_ids": json.dumps(comparison_ids),
                    "excluded_experiment_id": excluded_id,
                    "seed": SEED,
                }
            )

    # Preserve completed genes if a later reconstruction is interrupted.
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

results = pd.DataFrame(results)
summary = (
    results.groupby(["gene", "comparison_experiment_count"])["pearson_r"]
    .agg(["count", "mean", "std", "median"])
    .reset_index()
)
summary.to_csv(SUMMARY_PATH, index=False)
print(summary.to_string(index=False))
print(f"Saved results to {OUTPUT_PATH}")
print(f"Saved summary to {SUMMARY_PATH}")
