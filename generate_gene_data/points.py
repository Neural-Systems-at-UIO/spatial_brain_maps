import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyNutil.io.read_and_write import load_quint_json, write_points_to_meshview
from PyNutil.processing.transformations import image_to_atlas_space
from .utilities.path_utils import (
    id_to_data_path,
    id_to_quint_path,
    metadata
)
from .utilities.alignment_functions import load_warped_image


def value_to_rgb(v, vmin, vmax, cmap="viridis"):
    n = np.clip((v - vmin) / (vmax - vmin), 0, 1)
    rgba = np.array(plt.get_cmap(cmap)(n)) * 255
    return np.round(rgba).astype(int)[:3].tolist()


def section_index_to_atlas(
    section, resolution, cutoff, image_folder, affine_folder, nonlinear_folder, mode
):
    fname = os.path.splitext(os.path.basename(section["filename"]))[0]
    img = load_warped_image(
        fname,
        image_folder,
        affine_folder,
        nonlinear_folder,
        resolution,
        mode,
        int(section["filename"].split("/")[0]),
    )
    if img is None:
        return None, None

    pts = image_to_atlas_space(img, section["anchoring"]).reshape(-1, 3)
    vals = img.flatten()
    m = vals > cutoff
    return vals[m], pts[m]


def write_values_to_meshview(values, points, outfn, cmap="plasma"):
    u = np.unique(values)
    rgb = np.array([value_to_rgb(v, 0, 255, cmap) for v in u])
    df = pd.DataFrame(
        {
            "idx": u.tolist(),
            "name": u.tolist(),
            "r": rgb[:, 0],
            "g": rgb[:, 1],
            "b": rgb[:, 2],
        }
    )
    write_points_to_meshview(points, values, outfn, df)


def id_to_points(
    exp_id, image_folder, reg_folder, resolution=10, cutoff=0, mode="expression"
):
    aff = os.path.join(reg_folder, "affine_registration_files")
    non = os.path.join(reg_folder, "nonlinear_registration_files")
    slices = load_quint_json(
        id_to_quint_path(exp_id, reg_folder), propagate_missing_values=False
    )["slices"]
    slices = [s for s in slices if s["filename"].split("/")[0] == str(exp_id)]
    if image_folder:
        imgp = id_to_data_path(exp_id, image_folder)
    else:
        imgp = None
    affp = id_to_data_path(exp_id, aff)
    nonp = id_to_data_path(exp_id, non)

    all_vals, all_pts = [], []
    for s in slices:
        v, p = section_index_to_atlas(s, resolution, cutoff, imgp, affp, nonp, mode)
        if v is not None:
            all_vals.append(v)
            all_pts.append(p)

    return np.concatenate(all_vals), np.concatenate(all_pts)


def gene_to_points(
    gene,
    image_folder,
    reg_folder,
    cutoff=0,
    mode="expression",
    resolution=10,
    sleep_state="Nothing",
):
    md = metadata
    md = md[md["sleep_state"] == sleep_state]
    ids = md[md["gene"].str.lower() == gene.lower()].experiment_id.values
    vs, ps = zip(
        *(
            id_to_points(i, image_folder, reg_folder, resolution, cutoff, mode)
            for i in ids
        )
    )
    return np.concatenate(vs), np.concatenate(ps)
