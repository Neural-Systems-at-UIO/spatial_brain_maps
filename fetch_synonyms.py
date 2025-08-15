#!/usr/bin/env python3
"""Fetch gene synonyms, descriptions, Ensembl IDs and save counts."""
import json, gzip
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
SPECIES      = "mus_musculus"
METADATA_CSV = Path("../spatial_brain_maps/metadata/metadata.csv")
OUTPUT_DIR   = Path("data")

# ─── GENE INFO ─────────────────────────────────────────────────────────────────
def load_metadata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def fetch_gene_list(metadata: pd.DataFrame) -> list[str]:
    return metadata["gene"].unique().tolist()

def fetch_synonyms_and_desc(session: requests.Session, symbol: str):
    url = f"https://rest.ensembl.org/xrefs/name/{SPECIES}/{symbol}"
    r = session.get(url, headers={"Content-Type": "application/json"})
    if not r.ok: return None, None
    data = r.json()
    syns = {s for e in data for s in e.get("synonyms", [])}
    desc = data[0]["description"] if data else ""
    return syns, desc

def fetch_ensembl_id(session: requests.Session, symbol: str) -> str|None:
    url = f"https://rest.ensembl.org/xrefs/symbol/{SPECIES}/{symbol}"
    r = session.get(url, headers={"Content-Type": "application/json"})
    if not r.ok: return None
    res = r.json()
    return res[0]["id"] if res else None

def build_gene_info(metadata: pd.DataFrame) -> dict:
    sess = requests.Session()
    genes = fetch_gene_list(metadata)
    info = {"gene_name":[], "gene_description":[], "synonyms":[], "ensembl_ids":[]}
    for g in tqdm(genes, desc="Fetching Ensembl info"):
        syns, desc = fetch_synonyms_and_desc(sess, g)
        eid = fetch_ensembl_id(sess, g)
        info["gene_name"].append(g)
        info["gene_description"].append(desc or "")
        info["synonyms"].append(list(syns) if syns else [])
        info["ensembl_ids"].append(eid)
    return info

def add_counts_and_sort(info: dict, metadata: pd.DataFrame) -> dict:
    counts = [(metadata["gene"]==g).sum() for g in info["gene_name"]]
    info["number_of_animals"] = counts
    if "Nothing" in info["gene_name"]:
        idx = info["gene_name"].index("Nothing")
        for v in info.values(): v.pop(idx)
    rows = list(zip(*[info[k] for k in info]))
    rows.sort(key=lambda x: x[-1], reverse=True)
    for i, k in enumerate(info):
        info[k] = [r[i] for r in rows]
    return info

def save_json_gz(obj, path: Path, **kw):
    path.parent.mkdir(exist_ok=True, parents=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, **kw)

def main():
    meta = load_metadata(METADATA_CSV)
    info = build_gene_info(meta)
    info = add_counts_and_sort(info, meta)
    save_json_gz(info, OUTPUT_DIR/"gene_data_counts.json.gz", indent=4)

if __name__=="__main__":
    main()