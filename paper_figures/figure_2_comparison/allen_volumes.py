"""This script shows you how to create the volumes with the registrations from the Allen API."""
from spatial_brain_maps.utilities.path_utils import metadata
import requests
import os
import requests
import zipfile
from typing import List, Optional
import SimpleITK as sitk
import numpy as np
import nrrd

def download_expression_grid(
    section_id: int,
    include: Optional[List[str]] = None,
    out_dir: str = ".",
    timeout: int = 60,
) -> str:
    """
    Download 3-D expression grid data (zip) from Allen API.
    Returns path to the saved .zip file.
    """
    base = "http://api.brain-map.org/grid_data/download"
    params = {}
    if include:
        params["include"] = ",".join(include)

    url = f"{base}/{section_id}"
    resp = requests.get(url, params=params, stream=True, timeout=timeout)
    resp.raise_for_status()

    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{section_id}.zip")
    with open(fn, "wb") as fp:
        for chunk in resp.iter_content(1024 * 1024):
            fp.write(chunk)
    return fn

def load_mhd_volume(mhd_path: str):
    """Read the .mhd/.raw volume into a NumPy array."""
    # this will automatically look for the .raw named in the header
    img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(img)  # shape = [z, y, x]
    return img, arr

genes = [
    "Heatr5b",
    "Satb1",
    "Cacna1g",
    "Cap1"
]
counts = metadata[metadata['gene'].isin(genes)]['gene'].value_counts()
counts = dict(counts)
for gene in genes:
    subset = metadata[metadata['gene'] == gene].copy()
    for experiment_id in subset['experiment_id']:
        zip_path = download_expression_grid(
            experiment_id,
            include=["energy"],
            out_dir=f"outputs/{gene}/"
        )
        print("saved:", zip_path)
        # unzip the downloaded .zip into a same‐named folder
        with zipfile.ZipFile(zip_path, "r") as zf:
            extract_dir = os.path.splitext(zip_path)[0]
            os.makedirs(extract_dir, exist_ok=True)
            zf.extractall(extract_dir)
        print("extracted to:", extract_dir)
    
for gene in genes:
    vol_list = []
    subset = metadata[metadata['gene'] == gene].copy()
    for experiment_id in subset['experiment_id']:
        mhd_file = f"outputs/{gene}/{experiment_id}/energy.mhd"
        img, volume = load_mhd_volume(mhd_file)
        vol_list.append(volume)
    out = np.mean(vol_list, axis=0)
    nrrd.write(f"outputs/average_allen_{gene}.nrrd",out)


