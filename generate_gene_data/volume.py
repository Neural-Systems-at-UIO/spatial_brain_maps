import brainglobe_atlasapi
import os
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from .utilities.path_utils import (
    metadata,
    id_to_data_path,
    id_to_quint_path,
)
from PyNutil.io.read_and_write import load_quint_json
from .points import load_warped_image  # reuse that logic
from .utilities.center_images_from_alignment import (
    perfect_image,
    generate_square_alignment,
)
import cv2
from .utilities.generate_target_slice import generate_target_coordinates
from .utilities.nearestNDInterpolator import NearestNDInterpolator


def id_to_volume(
    exp_id,
    image_folder,
    reg_folder,
    resolution=25,
    mode="expression",
    return_frequencies=False,
    missing_fill=np.nan,
    do_interpolation=False,
    k=5
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

    sf = 25 / resolution
    shape = (np.array([11400, 14150, 8000]) / resolution).astype(int)
    gv = np.zeros(shape)
    fv = np.zeros(shape)

    for s in slices:
        fname = os.path.splitext(os.path.basename(s["filename"]))[0]
        anch = np.array(s["anchoring"]) * sf
        img = load_warped_image(fname, imgp, affp, nonp, resolution, mode, exp_id)
        if img is None:
            continue

        ip, _ = perfect_image(img, anch, resolution)
        ip = cv2.resize(ip, (shape[0], shape[2]), interpolation=cv2.INTER_AREA)
        sal = generate_square_alignment(anch, shape, resolution)
        coords = generate_target_coordinates(sal, shape)

        gv[coords] += ip.flatten()
        fv[coords] += 1
    if missing_fill != 0:
        gv[fv == 0] = missing_fill
    if do_interpolation:
        vol = interpolate(gv, fv, k=k, resolution=resolution)
    return (gv, fv) if return_frequencies else gv


def gene_to_volume(
    gene,
    image_folder=None,
    reg_folder=None,
    mode="expression",
    resolution=25,
    return_frequencies=False,
    missing_fill=np.nan,
    do_interpolation=True,
    k=5,
    sleep_state="Nothing",
):
    meta_df = metadata[metadata["sleep_state"] == sleep_state]
    ids = meta_df[meta_df["gene"].str.lower() == gene.lower()].experiment_id.values
    ids = metadata[metadata["gene"].str.lower() == gene.lower()].experiment_id.values
    shape = (np.array([11400, 14150, 8000]) / resolution).astype(int)
    gv = np.zeros(shape)
    fv = np.zeros(shape)

    for i in ids:
        vol, freq = id_to_volume(
            i,
            image_folder,
            reg_folder,
            resolution=resolution,
            mode=mode,
            return_frequencies=True,
            missing_fill=0,
            do_interpolation = do_interpolation,
            k=k
        )

        gv = gv + vol
        fv = fv + freq
    if not do_interpolation:
        mask = fv != 0
        gv[mask] = gv[mask] / fv[mask]
    else:
        gv = gv / len(ids)
    if (not do_interpolation) & (missing_fill != 0):
        gv[fv == 0] = missing_fill

    # reorient both to bg space before returning
    gv = gv[::-1, ::-1, ::-1].transpose(1, 2, 0)
    fv = fv[::-1, ::-1, ::-1].transpose(1, 2, 0)
    return (gv, fv) if return_frequencies else gv


def interpolate(gv, fv, k, resolution):
    atlas = brainglobe_atlasapi.BrainGlobeAtlas("ccfv3augmented_mouse_10um").annotation
    atlas = np.transpose(atlas, [2, 0, 1])[::-1, ::-1, ::-1]
    atlas_sh = np.array(atlas.shape)
    tgt_sh = atlas_sh * (10 / resolution)
    sf = tgt_sh / atlas_sh
    at = zoom(atlas, sf, order=0)
    mask, valid = at != 0, fv != 0
    fit, fill = mask & valid, mask & ~valid
    grid = np.mgrid[0 : tgt_sh[0], 0 : tgt_sh[1], 0 : tgt_sh[2]]
    pts = grid.reshape((3, -1)).T
    nn1 = NearestNDInterpolator(pts[fit.flatten()], gv[fit])
    out = gv.copy()
    out[fill] = nn1(pts[fill.flatten()], k=k)
    nn2 = NearestNDInterpolator(pts[fill.flatten()], out[fill])
    out[fit] = nn2(pts[fit.flatten()], k=k)
    return out


def write_nifti(volume, resolution, output_path, origin_offsets=None):
    if origin_offsets is None:
        """
        This is set to be compatible with Siibra explorer
        """
        origin_offsets = np.array([-5737.5, -7300, -4037.5])
    dims = volume.shape
    affine = np.eye(4)
    affine[:3, :3] *= resolution
    affine[:3, 3] = -0.5 * np.array(dims) * resolution + origin_offsets
    img = nib.Nifti1Image(volume.astype(np.uint8), affine)
    img.set_qform(affine, code=1)
    img.header["xyzt_units"] = 3
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path + ".nii.gz")
