#!/usr/bin/env python3
"""Compute search metrics from gene volumes in BrainGlobe array orientation."""

import argparse
import concurrent.futures
import gzip
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
METADATA_CSV = ROOT / "metadata/metadata.csv"
OUTPUT_DIR = ROOT / "data"
METRICS = ("intensity", "specificity", "expr_pct", "expr_spec")


def build_region_descendants(atlas, vol: np.ndarray) -> dict[int, list[int]]:
    """Include every hierarchy region with labelled voxels in its subtree.

    Parent regions (e.g. Dentate gyrus) need not have their own voxel label.
    Include a region's own label as well as all of its labelled descendants.
    """
    region_set = set(np.unique(vol).tolist()) - {0}
    unknown = region_set - set(atlas.structures)
    if unknown:
        raise ValueError(f"Annotation labels absent from atlas hierarchy: {unknown}")
    desc_map = {}
    for rid in sorted(atlas.structures):
        if not rid:
            continue
        descendants = set(atlas.hierarchy.expand_tree(int(rid))) | {rid}
        present = sorted(descendants & region_set)
        if present:
            desc_map[rid] = present
    return desc_map


def annotation_for_orientation(annotation, orientation="brainglobe"):
    """Current gene_to_volume returns native BrainGlobe arrays.

    The legacy option is only for volumes produced before that reorientation
    was added to generate_gene_data/volume.py. Never flatten mismatched shapes.
    """
    if orientation == "brainglobe":
        return annotation
    if orientation == "legacy":
        return annotation.transpose(2, 0, 1)[::-1, ::-1, ::-1]
    raise ValueError(f"Unknown volume orientation: {orientation}")


def validate_volume(image, shape, path, resolution=None):
    if image.shape != tuple(shape):
        raise ValueError(
            f"{path}: gene volume shape {image.shape} does not match annotation "
            f"shape {tuple(shape)}. Check --volume-orientation and atlas resolution."
        )
    if resolution is not None:
        spacing = np.array(image.header.get_zooms())
        units = image.header.get_xyzt_units()[0]
        spacing *= {"meter": 1e6, "mm": 1e3, "micron": 1}.get(units, 1)
        if not np.allclose(spacing, resolution):
            raise ValueError(
                f"{path}: voxel spacing {spacing} does not match {resolution} um"
            )


def compute_region_stats(meta, desc_map, vol, volume_dir, max_workers=2):
    nonz = vol.ravel() != 0
    # Compact sparse atlas IDs before bincount: some IDs are hundreds of millions.
    labels, base = np.unique(vol.ravel()[nonz], return_inverse=True)
    counts = np.bincount(base)
    region_ids = [rid for rid, descendants in desc_map.items() if rid and descendants]
    indices = [np.searchsorted(labels, desc_map[rid]) for rid in region_ids]
    region_counts = np.array([counts[d].sum() for d in indices])
    if not region_ids or np.any(region_counts == 0):
        raise ValueError("No labelled voxels for one or more search regions")
    stats = {kind: {rid: [] for rid in region_ids} for kind in METRICS}
    stats["gene_name"] = []
    gene_list = [g for g in meta["gene"].unique() if g != "Nothing"]
    if not gene_list:
        raise ValueError("No gene volumes match the filtered metadata")
    eps = 1e-8

    def worker(gene):
        path = Path(volume_dir) / f"{gene}.nii.gz"
        image = nib.load(path)
        validate_volume(image, vol.shape, path)
        values = np.asanyarray(image.dataobj).ravel()[nonz]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{path}: expression must be finite and non-negative")
        sums = np.bincount(base, weights=values, minlength=len(labels))
        expressed = np.bincount(base, weights=values > 0, minlength=len(labels))
        means = np.array([sums[d].sum() for d in indices]) / region_counts
        pcts = np.array([expressed[d].sum() for d in indices]) / region_counts
        # Retain the existing definition: each region's mean / sum of regional means.
        specificity = (means + eps) / (means.sum() + eps * len(means))
        pct_spec = (pcts + eps) / (pcts.sum() + eps * len(pcts))
        return gene, (means, specificity, pcts, pct_spec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # map preserves gene order, keeping exported names paired with their values.
        results = executor.map(worker, gene_list)
        for gene, values in tqdm(results, desc="Region stats", total=len(gene_list)):
            stats["gene_name"].append(gene)
            for kind, arr in zip(METRICS, values):
                for rid, value in zip(region_ids, arr):
                    stats[kind][rid].append(float(value))
    return stats


def save_json_gz(obj, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    payload = json.dumps(obj, allow_nan=False).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0))


def export_region_stats(stats, atlas, output_dir=OUTPUT_DIR):
    gene_names = stats["gene_name"]
    for kind in METRICS:
        for rid, values in stats[kind].items():
            name = re.sub(r"[\\/]", "_", atlas.structures[int(rid)]["name"])
            save_json_gz(
                dict(zip(gene_names, values, strict=True)),
                Path(output_dir) / "metrics" / f"{name}_{kind}.json.gz",
            )
    names = [atlas.structures[rid]["name"] for rid in stats["intensity"]]
    save_json_gz(names, Path(output_dir) / "structure_names.json.gz")


def load_metadata(metadata_path, volume_dir):
    meta = pd.read_csv(metadata_path)
    for field, value in {
        "sleep_state": "Nothing",
        "plane_of_section": "coronal",
        "treatment": "ISH",
        "age": "P56",
    }.items():
        meta = meta[meta[field] == value]
    meta = meta[meta["gene"].notna() & (meta["gene"] != "Nothing")].copy()
    # Windows drives encode '*' as a private-use character. Use actual file stems.
    available = {p.name[:-7]: p for p in Path(volume_dir).glob("*.nii.gz")}

    def file_gene(gene):
        return gene if gene in available else gene.replace("*", "\uf02a")

    meta["gene"] = meta["gene"].map(file_gene)
    return meta[meta["gene"].isin(available)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--atlas", default="ccfv3augmented_mouse_25um")
    parser.add_argument(
        "--volume-orientation", choices=("brainglobe", "legacy"), default="brainglobe"
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.volume_dir.is_dir():
        parser.error(f"Volume directory does not exist: {args.volume_dir}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    meta = load_metadata(args.metadata, args.volume_dir)
    if meta.empty:
        parser.error("No gene volumes match the filtered metadata")

    from brainglobe_atlasapi import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(args.atlas)
    annotation = annotation_for_orientation(atlas.annotation, args.volume_orientation)
    print(
        f"Checking {meta['gene'].nunique()} genes against annotation {annotation.shape}",
        flush=True,
    )
    # Check all headers before doing expensive work or writing any output.
    for gene in meta["gene"].unique():
        path = args.volume_dir / f"{gene}.nii.gz"
        validate_volume(nib.load(path), annotation.shape, path, atlas.resolution)
    descendants = build_region_descendants(atlas, annotation)
    stats = compute_region_stats(
        meta, descendants, annotation, args.volume_dir, args.workers
    )
    export_region_stats(stats, atlas, args.output_dir)
    print(
        f"Exported {len(descendants)} regions and {len(stats['gene_name'])} genes",
        flush=True,
    )


if __name__ == "__main__":
    main()
