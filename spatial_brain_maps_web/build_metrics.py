#!/usr/bin/env python3
"""Compute region metrics from annotation volume and gene volumes."""
import gzip, re
from pathlib import Path
import brainglobe_atlasapi
from glob import glob
import numpy as np
import pandas as pd
import nibabel as nib
import json
from tqdm import tqdm
import concurrent.futures, os

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
METADATA_CSV = Path("../spatial_brain_maps/metadata/metadata.csv")
NIIGZ_DIR      = Path("../outputs/gene_volumes")
OUTPUT_DIR     = Path("data")

# ─── BUILD DESCENDANTS VIA BrainGlobeAtlas ────────────────────────────────────────
def build_region_descendants(atlas, vol: np.ndarray) -> dict[int, list[int]]:
    """
    For each label in vol (except 0), expand via atlas.hierarchy
    and keep only those descendants also present in vol.
    """
    region_set = set(np.unique(vol).tolist()) - {0}
    desc_map = {}
    for rid in sorted(region_set):
        all_desc = list(atlas.hierarchy.expand_tree(int(rid)))
        desc_map[rid] = [d for d in all_desc if d in region_set]
    return desc_map

# ─── STATS COMPUTATION ──────────────────────────────────────────────────────────
atlas = brainglobe_atlasapi.BrainGlobeAtlas(
    "ccfv3augmented_mouse_25um"
).annotation
ANNOT_VOL = np.transpose(atlas, [2, 0, 1])[::-1, ::-1, ::-1]

def compute_region_stats(meta, desc_map, vol):
    nonz = vol.ravel() != 0
    base = vol.ravel()[nonz]
    # only regions with descendants
    region_ids = [rid for rid, d in desc_map.items() if rid and d]
    # init (remove r_squared)
    stats = {
        k: {rid: [] for rid in region_ids}
        for k in (
            "intensity",
            "specificity",
            "expr_pct",
            "expr_spec",
            # "r_squared",
        )
    }
    stats["gene_name"] = []
    gene_list = [g for g in meta["gene"].unique() if g != "Nothing"]
    eps = 1e-8

    def _worker(g):
        nim = nib.load(NIIGZ_DIR / f"{g}.nii.gz")
        gvol = nim.get_fdata().ravel()[nonz]
        # sums & counts per label
        sums   = np.bincount(base, weights=gvol, minlength=base.max()+1)
        counts = np.bincount(base, minlength=base.max()+1)
        # per-region mean
        means = {
            rid: sums[d].sum() / counts[d].sum()
            if counts[d].sum() else 0.0
            for rid, d in desc_map.items() if rid in region_ids
        }
        # specificity
        arr = np.array(list(means.values()))
        spec = (arr + eps) / (arr.sum() + eps * len(arr))
        # percentage expressed
        mask_expr = (gvol > 0).astype(int)
        expr = np.bincount(base, weights=mask_expr, minlength=base.max()+1)
        pcts = {
            rid: expr[d].sum() / counts[d].sum()
            if counts[d].sum() else 0.0
            for rid, d in desc_map.items() if rid in region_ids
        }
        pct_arr  = np.array(list(pcts.values()))
        pct_spec = (pct_arr + eps) / (pct_arr.sum() + eps * len(pct_arr))
        # R² calculation removed
        # r2_map = {}
        # for i, rid in enumerate(region_ids):
        #     mask = np.isin(base, desc_map[rid]).astype(int)
        #     if mask.sum() > 1:
        #         c = np.corrcoef(gvol, mask)[0, 1]
        #         r2_map[rid] = float(c**2) if not np.isnan(c) else 0.0
        #     else:
        #         r2_map[rid] = 0.0

        return {
            "gene":  g,
            "means": means,
            "spec":  dict(zip(region_ids, spec)),
            "pcts":  pcts,
            "pctsp": dict(zip(region_ids, pct_spec)),
            # "r2":    r2_map,
        }

    # run in a pool (preserves submission order)
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
        futures = [exe.submit(_worker, g) for g in gene_list]
        for f in tqdm(futures, desc="Region stats", total=len(futures)):
            res = f.result()
            stats["gene_name"].append(res["gene"])
            for rid in region_ids:
                stats["intensity"][rid].append(res["means"][rid])
                stats["specificity"][rid].append(res["spec"][rid])
                stats["expr_pct"][rid].append(res["pcts"][rid])
                stats["expr_spec"][rid].append(res["pctsp"][rid])
                # stats["r_squared"][rid].append(res["r2"][rid])

    return stats

def save_json_gz(obj, path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with gzip.open(path, "wt") as f: json.dump(obj, f)

def export_region_stats(stats, atlas):
    for kind,d in stats.items():
        if kind=="gene_name": continue
        df = pd.DataFrame(d)
        for rid in df.columns.drop("gene_name"):
            name = atlas.hierarchy.get_region(int(rid)).name
            name = re.sub(r'[\\/]', '_', name)
            save_json_gz(dict(zip(df["gene_name"], df[rid])),
                         OUTPUT_DIR/f"metrics/{name}_{kind}.json.gz")

meta = pd.read_csv(METADATA_CSV)
meta = meta[meta['sleep_state'] == 'Nothing']
genes = ['.'.join(os.path.basename(i).split('.')[:-2]) for i in glob("../outputs/gene_volumes/*.nii.gz")]

meta[~meta['gene'].isin(genes)]['gene'].value_counts()
atlas = brainglobe_atlasapi.BrainGlobeAtlas("ccfv3augmented_mouse_25um")
desc_map = build_region_descendants(atlas, ANNOT_VOL)
stats = compute_region_stats(meta, desc_map, ANNOT_VOL)
export_region_stats(stats, atlas)
valid = [atlas.hierarchy.get_region(int(rid)).name
            for rid,d in desc_map.items() if rid and d]
save_json_gz(valid, OUTPUT_DIR/"structure_names.json.gz")

