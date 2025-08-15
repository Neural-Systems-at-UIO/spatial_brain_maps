import os
import numpy as np
import itertools
from glob import glob
import nibabel as nib
import PyNutil
import ants
import math
import cv2
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
from skimage.morphology import thin
import pandas as pd
import csv
from misc.visualign_deformations import transform_vec, triangulate

# full path to where your per‐section ANTs affine mats live
AFFINE_FOLDER = "datafiles/raters/pipeline_registrations/affine/"
atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
raw_outline3d = find_boundaries(annot, mode="inner", connectivity=annot.ndim)
template = np.transpose(atlas.reference, (2, 0, 1))[::-1, ::-1, ::-1]
template = template / template.max()


def generate_target_slice(orientation, atlas):
    """
    Generate a 2D slice from a 3D atlas based on orientation vectors.

    Args:
        orientation (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        atlas (ndarray): 3D atlas volume.

    Returns:
        ndarray: 2D slice extracted from the atlas.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = orientation
    slice_width = np.floor(math.hypot(ux, uy, uz)).astype(int) + 1
    slice_height = np.floor(math.hypot(vx, vy, vz)).astype(int) + 1
    data = np.zeros((slice_width, slice_height), dtype=np.uint32).flatten()
    xdim, ydim, zdim = atlas.shape

    y_values = np.arange(slice_height)
    x_values = np.arange(slice_width)

    hx = ox + vx * (y_values / slice_height)
    hy = oy + vy * (y_values / slice_height)
    hz = oz + vz * (y_values / slice_height)

    wx = ux * (x_values / slice_width)
    wy = uy * (x_values / slice_width)
    wz = uz * (x_values / slice_width)

    lx = np.floor(hx[:, None] + wx).astype(int)
    ly = np.floor(hy[:, None] + wy).astype(int)
    lz = np.floor(hz[:, None] + wz).astype(int)

    valid_indices = (
        (0 <= lx) & (lx < xdim) & (0 <= ly) & (ly < ydim) & (0 <= lz) & (lz < zdim)
    ).flatten()

    lxf = lx.flatten()
    lyf = ly.flatten()
    lzf = lz.flatten()

    valid_lx = lxf[valid_indices]
    valid_ly = lyf[valid_indices]
    valid_lz = lzf[valid_indices]

    atlas_slice = atlas[valid_lx, valid_ly, valid_lz]
    data[valid_indices] = atlas_slice

    data_im = data.reshape((slice_height, slice_width))
    return data_im


def calculate_affine(source_points, destination_points):
    # Add a fourth coordinate of 1 to each point
    source_points = np.hstack((source_points, np.ones((source_points.shape[0], 1))))
    destination_points = np.hstack(
        (destination_points, np.ones((destination_points.shape[0], 1)))
    )
    affine_matrix, _, _, _ = np.linalg.lstsq(
        source_points, destination_points, rcond=None
    )
    return affine_matrix.T


def read_ants_affine(affine_path):
    if not os.path.exists(affine_path):
        return None
    ants_transform = ants.read_transform(affine_path)
    base_points = np.array([[0, 0], [0, 1], [1, 0]])
    transformed_points = np.array(
        [ants_transform.apply_to_point(p) for p in base_points]
    )
    affine_matrix = calculate_affine(base_points, transformed_points)
    return affine_matrix


def apply_affine_to_points(affine_matrix, points):
    # Pre-shift the points into padded image coordinates
    points_homogeneous = np.column_stack((points, np.ones(len(points))))
    warped_points = (affine_matrix @ points_homogeneous.T).T
    return warped_points[:, :2]


# ─── helper: apply non-linear DF to 2D points ───────────────────────────────
def apply_nonlinear_to_points(deformation_field, output_shape, points):
    """
    deformation_field: H x W x 2 numpy array (DX, DY at each voxel)
    points:            N x 2 array of (x, y) coordinates
    returns:           N x 2 array of warped points
    """
    coords = [points[:, 1], points[:, 0]]  # map_coordinates expects (row, col)
    target_h, target_w = output_shape
    scale_y = target_h / deformation_field.shape[0]
    scale_x = target_w / deformation_field.shape[1]
    resized_deformation = zoom(deformation_field, (scale_y, scale_x, 1), order=1)
    dx = resized_deformation[:, :, 0].flatten()
    dy = resized_deformation[:, :, 1].flatten()
    return points - np.stack((dx, dy), axis=1)


# ─── Core geometry routines ────────────────────────────────────────────────────
def find_plane_equation(plane_params):
    a, b, c = (
        np.array(plane_params[:3], float),
        np.array(plane_params[3:6], float),
        np.array(plane_params[6:9], float),
    )
    normal = np.cross(b, c) / 9.0
    k = -np.dot(a, normal)
    return normal, k


def get_angle(plane_params, direction):
    normal, k = find_plane_equation(plane_params)
    plane_coords = plane_params.copy()
    for i in range(3):
        plane_coords[i + 3] += plane_coords[i]
        plane_coords[i + 6] += plane_coords[i]
    if direction == "ML":
        a = plane_coords[0:2]
        linear_point = (
            ((plane_coords[0] - 100) * normal[0]) + ((plane_coords[2]) * normal[2])
        ) + k
        depth = -(linear_point / normal[1])
        b = np.array((plane_coords[0] - 100, depth))
        c = b + [100, 0]
    if direction == "DV":
        a = plane_coords[1:3]
        linear_point = (
            ((plane_coords[0]) * normal[0]) + ((plane_coords[2] - 100) * normal[2])
        ) + k
        depth = -(linear_point / normal[1])
        b = np.array((depth, plane_coords[2] - 100))
        c = b + [0, 100]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    if direction == "ML":
        if b[1] > a[1]:
            angle *= -1
    if direction == "DV":
        if b[0] < a[0]:
            angle *= -1
    return angle


def normalised_grid(alignment, num_x, num_z):
    normal, k = find_plane_equation(alignment)
    xv, zv = np.meshgrid(np.arange(num_x), np.arange(num_z))
    y_ = xv * normal[0] + zv * normal[2] + k
    return xv, -(y_ / normal[1]), zv


def xyz_to_pix(coordinates_3d, h, w, alignment):
    O, U, V = alignment[:3], alignment[3:6], alignment[6:9]
    pts = np.vstack(coordinates_3d)
    b = pts - O[:, None]
    M = np.stack([U, V], axis=1)
    sol, *_ = np.linalg.lstsq(M, b, rcond=None)
    t, s = sol
    return t * w, s * h


def pix_to_xyz(px, py, h, w, alignment):
    xfrac, yfrac = px / w, py / h
    O, U, V = alignment[:3], alignment[3:6], alignment[6:9]
    offs_u = xfrac[:, None] * U[None, :]
    offs_v = yfrac[:, None] * V[None, :]
    pts = O[None, :] + offs_u + offs_v
    return pts[:, 0], pts[:, 1], pts[:, 2]


def flatten_grid(nx, ny):
    # Create and flatten a mesh grid for simpler usage
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
    return grid_x.flatten(), grid_y.flatten()


# ─── I/O ───────────────────────────────────────────────────────────────────────
def get_slice_json(path, section_nr):
    data = PyNutil.io.read_and_write.load_quint_json(path)
    return next(s for s in data["slices"] if s["nr"] == section_nr)


def load_alignments(
    section_nr, human_files, ds_files, aba_files, ds_y_shift=24, ds_basic_files=None
):
    aligns = {}
    markers = {}
    # 1) load humans
    for name, paths in human_files.items():
        sl = get_slice_json(paths[0], section_nr)
        aligns[name] = np.array(sl["anchoring"], float)
        markers[name] = np.array(sl["markers"]) if "markers" in sl else None
    height, width = sl["height"], sl["width"]
    # 2) raw DeepSlice from ds_human_affine files
    sl = get_slice_json(ds_files[0], section_nr)
    a = np.array(sl["anchoring"], float)
    a[1] += ds_y_shift
    aligns["Our Pipeline"] = a

    # 4) ABA registrations (apply same ds offset)
    sl_aba = get_slice_json(aba_files[0], section_nr)
    a_aba = np.array(sl_aba["anchoring"], float)
    a_aba[1] += ds_y_shift
    aligns["ABA"] = a_aba

    return aligns, markers, height, width


# ─── Core error computations ─────────────────────────────────────────────────
def compute_error(
    test_name,
    test_align,
    ref_aligns,
    annot,
    h,
    w,
    section_nr=None,
    brain_id=None,
    test_markers=None,
    ref_markers=None,
):
    # project dense grid from mean(ref_aligns)
    ### For the affine and nonlinear we have to have the shapes that coresspond to the image size
    sec4 = str(section_nr).zfill(4)
    aff_pat = os.path.join(
        AFFINE_FOLDER, brain_id, "*", f"*_s{sec4}_SyN_affineTransfo.mat"
    )
    mats = glob(aff_pat)
    nl_path = (
        mats[0]
        .replace("_SyN_affineTransfo.mat", "_SyN_nonLinearDf.nii.gz")
        .replace("affine", "nonlin")
    )
    if os.path.exists(nl_path):
        nl_im = nib.load(nl_path).get_fdata()
        ny, nx = nl_im.shape[0], nl_im.shape[1]
    px, py = flatten_grid(nx, ny)
    if test_name == "Our Pipeline":
        add = nl_im[py, px].squeeze()
        py = py - add[:, 0]
        px = px - add[:, 1]
    if test_markers is not None:
        triangulation = triangulate(w, h, test_markers)
        px = (px / nx) * w
        py = (py / ny) * h
        px, py = transform_vec(triangulation, px, py)
        px = (px / w) * nx
        py = (py / h) * ny
    mean_ref = np.mean(ref_aligns, axis=0)
    tempx, tempy = flatten_grid(nx, ny)
    if ref_markers is not None:
        aggregate_x = []
        aggregate_y = []
        for ref_m in ref_markers:
            tempx_copy = tempx.copy()
            tempy_copy = tempy.copy()
            if ref_m is not None:
                triangulation = triangulate(w, h, ref_m)
                tempx_copy = (tempx_copy / nx) * w
                tempy_copy = (tempy_copy / ny) * h
                tempx_copy, tempy_copy = transform_vec(
                    triangulation, tempx_copy, tempy_copy
                )
                tempx_copy = (tempx_copy / w) * nx
                tempy_copy = (tempy_copy / h) * ny
                aggregate_x.append(tempx_copy)
                aggregate_y.append(tempy_copy)
        tempx = np.array(aggregate_x).mean(axis=0)
        tempy = np.array(aggregate_y).mean(axis=0)

    gtx, gty, gtz = pix_to_xyz(tempx, tempy, ny, nx, mean_ref)
    # now lift back into 3D
    x3, y3, z3 = pix_to_xyz(px, py, ny, nx, test_align)

    # plt.imshow((at / at.max()).astype(np.uint8))
    # plt.show()
    diffs = np.vstack((x3, y3, z3)).T - np.vstack((gtx, gty, gtz)).T

    x3[x3 < 0] = 0
    y3[y3 < 0] = 0
    z3[z3 < 0] = 0
    x3[x3 >= template.shape[0]] = template.shape[0] - 1
    y3[y3 >= template.shape[1]] = template.shape[1] - 1
    z3[z3 >= template.shape[2]] = template.shape[2] - 1
    at = raw_outline3d[x3.astype(int), y3.astype(int), z3.astype(int)]
    at = at.reshape(ny, nx)
    at = thin(at)
    # Insert reading of image from file
    im_path = glob(f"section_images/{brain_id}/thumbnails/*_s{sec4}.jpg")[0]
    image = cv2.imread(im_path)
    os.makedirs(f"plots/atlas_outlines//{brain_id}/{test_name}", exist_ok=True)
    image = cv2.resize(image, (at.shape[1], at.shape[0]))
    image[at != 0] = [0, 0, 255]
    cv2.imwrite(f"plots/atlas_outlines//{brain_id}/{test_name}/{sec4}.jpg", image)
    return np.mean(np.linalg.norm(diffs, axis=1))


def leave_one_out_humans(
    aligns, human_names, annot, h, w, section_nr, brain_id, markers
):
    """
    Returns {human_name: error vs mean(of the other 2)}.
    """
    errs = {}
    for name in human_names:
        refs = [aligns[n] for n in human_names if n != name]
        ref_m = [markers[n] for n in human_names if n != name]
        # pass the rater name as test_name
        errs[name] = compute_error(
            name,
            aligns[name],
            refs,
            annot,
            h,
            w,
            section_nr,
            brain_id,
            markers[name],
            ref_m,
        )
    return errs


def test_vs_all_humans(
    aligns,
    test_name,
    human_names,
    annot,
    h,
    w,
    section_nr=None,
    brain_id=None,
    markers=None,
):
    """
    For `test_name` rater, compute error vs all human raters except one (leave-one-out).
    Returns mean_error.
    """
    errors = []
    for excluded_name in human_names:
        refs = [aligns[n] for n in human_names if n != excluded_name]
        ref_m = [markers[n] for n in human_names if n != name]
        e = compute_error(
            test_name,
            aligns[test_name],
            refs,
            annot,
            h,
            w,
            section_nr=section_nr,
            brain_id=brain_id,
            test_markers=None,
            ref_markers=ref_m,
        )
        errors.append(e)

    return np.mean(errors)


def save_consolidated_results(results, filename):
    """
    Save all collected results to a CSV file with standardized columns.

    Args:
        results (list): List of dictionaries containing section data
        filename (str): Output CSV file path
    """
    # Extract all possible column names from the results
    all_keys = set()
    for entry in results:
        all_keys.update(entry.keys())

    # Define the fieldnames in a logical order
    fieldnames = [
        "brain_id",
        "section_nr",
        *sorted(k for k in all_keys if k.endswith("_error")),
        *sorted(k for k in all_keys if k.endswith("_ml")),
        *sorted(k for k in all_keys if k.endswith("_dv")),
        "human_ml_avg",
        "human_ml_std",
        "human_dv_avg",
        "human_dv_std",
    ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def create_group_atlas_overlay(
    aligns,
    markers,
    human_names,
    annot,
    raw_outline3d,
    template,
    H,
    W,
    section_nr,
    brain_id,
):
    """
    Create and save an atlas overlay image using the group average alignment and
    combined non-linear marker transformations from all human raters.

    Args:
        aligns (dict): Dictionary of alignments keyed by rater name.
        markers (dict): Dictionary of markers keyed by rater name.
        human_names (list): List of human rater names.
        annot (ndarray): Atlas annotation volume.
        raw_outline3d (ndarray): 3D outline volume from the atlas.
        template (ndarray): Atlas reference template.
        H, W (int): Height and width of the image.
        section_nr (int): Section number.
        brain_id (str): Brain identifier.
    """
    sec4 = str(section_nr).zfill(4)
    # Locate the NL deformation field to calculate size only
    aff_pat = os.path.join(
        AFFINE_FOLDER, brain_id, "*", f"*_s{sec4}_SyN_affineTransfo.mat"
    )
    mats = glob(aff_pat)
    if not mats:
        print(f"No affine matrices found for brain {brain_id} section {sec4}")
        return
    nl_path = (
        mats[0]
        .replace("_SyN_affineTransfo.mat", "_SyN_nonLinearDf.nii.gz")
        .replace("affine", "nonlin")
    )
    if os.path.exists(nl_path):
        nl_im = nib.load(nl_path).get_fdata()
        ny, nx = nl_im.shape[0], nl_im.shape[1]
    else:
        print(
            f"Nonlinear deformation field not found for brain {brain_id} section {sec4}"
        )
        return

    # Build a grid based on the NL dimensions
    tempx, tempy = flatten_grid(nx, ny)

    # If marker data are available, combine each rater's transformation similar to compute_error.
    aggregate_x = []
    aggregate_y = []
    for name in human_names:
        ref_m = markers.get(name)
        if ref_m is not None:
            # Using the same approach as in compute_error for each marker set:
            triangulation = triangulate(W, H, ref_m)
            # Scale grid to image coordinates, transform, then re-scale to NL dimensions
            tempx_copy = (tempx / nx) * W
            tempy_copy = (tempy / ny) * H
            tempx_copy, tempy_copy = transform_vec(
                triangulation, tempx_copy, tempy_copy
            )
            tempx_copy = (tempx_copy / W) * nx
            tempy_copy = (tempy_copy / H) * ny
            aggregate_x.append(tempx_copy)
            aggregate_y.append(tempy_copy)
    if aggregate_x:
        tempx = np.array(aggregate_x).mean(axis=0)
        tempy = np.array(aggregate_y).mean(axis=0)

    # Get the group average alignment (the "mean reference")
    group_align = np.mean([aligns[name] for name in human_names], axis=0)

    # Convert the adjusted 2D grid to 3D atlas coordinates using the group alignment
    gtx, gty, gtz = pix_to_xyz(tempx, tempy, ny, nx, group_align)

    # Clip coordinates to template range
    gtx[gtx < 0] = 0
    gty[gty < 0] = 0
    gtz[gtz < 0] = 0
    gtx[gtx >= template.shape[0]] = template.shape[0] - 1
    gty[gty >= template.shape[1]] = template.shape[1] - 1
    gtz[gtz >= template.shape[2]] = template.shape[2] - 1

    # Extract the atlas outline for these coordinates and reshape to (ny,nx)
    at = raw_outline3d[gtx.astype(int), gty.astype(int), gtz.astype(int)]
    at = at.reshape(ny, nx)
    at = thin(at)

    # Read the corresponding section image (expects a matching thumbnail)
    im_files = glob(f"section_images/{brain_id}/thumbnails/*_s{sec4}.jpg")
    if not im_files:
        print(f"No section image found for brain {brain_id} section {sec4}")
        return
    image = cv2.imread(im_files[0])
    image = cv2.resize(image, (at.shape[1], at.shape[0]))
    image[at != 0] = [0, 0, 255]  # Overlay in red

    out_dir = os.path.join("plots", "atlas_outlines", brain_id, "group_average_all")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sec4}.jpg")
    cv2.imwrite(out_path, image)


# Define expert and novice raters
expert_files = {
    "Expert 1": glob("datafiles/raters/experts/ingvild/*.json"),
    "Expert 2": glob("datafiles/raters/experts/sharon/*.json"),
    "Expert 3": glob("datafiles/raters/experts/simon/*.json"),
}
novice_files = {
    "Novice 1": glob("datafiles/raters/novices/signy/*.json"),
    "Novice 2": glob("datafiles/raters/novices/sophia/*.json"),
    "Novice 3": glob("datafiles/raters/novices/archana/*.json"),
}
human_files = {**expert_files, **novice_files}
expert_names = list(expert_files.keys())
novice_names = list(novice_files.keys())
human_names = list(human_files.keys())

ds_files = glob("datafiles/raters/pipeline_registrations/ds_human_affine/*.json")
aba_files = glob("datafiles/raters/ABA/*.json")

brain_ids = ["04-0180", "05-3097", "06-0262", "1966", "1984", "335-1118"]
ds_y_shift = 24
method_names = ["Our Pipeline", "ABA"]
rater_names = human_names + method_names


results = []

for brain in brain_ids:
    print(f"\n=== Brain: {brain} ===")

    hf_brain = {
        name: [
            p
            for p in human_files[name]
            if os.path.splitext(os.path.basename(p))[0] == brain
        ]
        for name in human_names
    }
    ds_brain = [
        p for p in ds_files if os.path.splitext(os.path.basename(p))[0] == brain
    ]
    aba_brain = [
        p for p in aba_files if os.path.splitext(os.path.basename(p))[0] == brain
    ]

    files_all = sum(hf_brain.values(), []) + ds_brain + aba_brain
    sec_sets = []
    for path in files_all:
        data = PyNutil.io.read_and_write.load_quint_json(path)
        sec_sets.append({s["nr"] for s in data["slices"]})
    section_nrs = sorted(set.intersection(*sec_sets))

    # Initialize accumulators for this brain
    brain_human_acc = {n: [] for n in human_names}
    our_pipeline_acc = []
    brain_aba_acc = []

    for sec in section_nrs:
        aligns, markers, H, W = load_alignments(
            sec, hf_brain, ds_brain, aba_brain, ds_y_shift
        )

        # Create entry for this section
        section_data = {"brain_id": brain, "section_nr": sec}

        # 1) Human leave-one-out errors
        human_errs = leave_one_out_humans(
            aligns, human_names, annot, H, W, sec, brain, markers
        )
        for name, err in human_errs.items():
            brain_human_acc[name].append(err)
            section_data[f"{name}_error"] = err

            # Store angles for humans
            ml_angle = get_angle(aligns[name], "ML")
            dv_angle = get_angle(aligns[name], "DV")
            section_data[f"{name}_ml"] = ml_angle
            section_data[f"{name}_dv"] = dv_angle

        # Calculate human averages
        human_ml = [section_data[f"{n}_ml"] for n in human_names]
        human_dv = [section_data[f"{n}_dv"] for n in human_names]
        section_data["human_ml_avg"] = np.mean(human_ml)
        section_data["human_ml_std"] = np.std(human_ml)
        section_data["human_dv_avg"] = np.mean(human_dv)
        section_data["human_dv_std"] = np.std(human_dv)

        # 2) Method errors
        methods = [("Our Pipeline", "Our Pipeline"), ("ABA", "ABA")]

        for method_key, method_name in methods:
            if method_name in aligns:
                # Compute error against all humans (n-1)
                mean_err = test_vs_all_humans(
                    aligns, method_name, human_names, annot, H, W, sec, brain, markers
                )

                # Store in appropriate accumulator
                if method_key == "Our Pipeline":
                    our_pipeline_acc.append(mean_err)
                elif method_key == "ABA":
                    brain_aba_acc.append(mean_err)

                # Store in section data
                section_data[f"{method_key}_error"] = mean_err

                # Store angles for methods
                ml_angle = get_angle(aligns[method_name], "ML")
                dv_angle = get_angle(aligns[method_name], "DV")
                section_data[f"{method_key}_ml"] = ml_angle
                section_data[f"{method_key}_dv"] = dv_angle
        create_group_atlas_overlay(
            aligns,
            markers,
            human_names,
            annot,
            raw_outline3d,
            template,
            H,
            W,
            sec,
            brain,
        )
        results.append(section_data)

    # Print brain summary
    print(f"\n=== Brain: {brain} summary ===")
    # Human raters
    for name in human_names:
        mean_err = np.mean(brain_human_acc[name])
        med_err = np.median(brain_human_acc[name])
        print(f"  {name:15} mean: {mean_err:.3f}, median: {med_err:.3f}")

    # Methods
    methods = [("Our Pipeline", our_pipeline_acc), ("ABA", brain_aba_acc)]
    for method, vals in methods:
        if vals:  # Only print if we have values
            mean_err = np.mean(vals)
            med_err = np.median(vals)
            print(f"  {method:15} mean: {mean_err:.3f}, median: {med_err:.3f}")


# Save consolidated results
save_consolidated_results(results, "consolidated_registration_results.csv")

# Print final summary across all brains
print("\n=== Final summary across all brains ===")

# Convert results to DataFrame for easier analysis
df = pd.DataFrame(results)

# Calculate average error for each rater across all brains
print("\nAverage error across all brains:")
for name in human_names:
    avg_error = df[f"{name}_error"].mean()
    std_error = df[f"{name}_error"].std()
    print(f"  {name:15} mean: {avg_error:.3f} ± {std_error:.3f}")

# Calculate average error for methods
for method in method_names:
    if f"{method}_error" in df.columns:
        avg_error = df[f"{method}_error"].mean()
        std_error = df[f"{method}_error"].std()
        print(f"  {method:15} mean: {avg_error:.3f} ± {std_error:.3f}")

# Group by brain_id to show average errors per brain
print("\nAverage errors by brain:")
for brain_id in brain_ids:
    brain_df = df[df["brain_id"] == brain_id]
    print(f"\n  Brain: {brain_id}")

    # Human raters
    for name in human_names:
        brain_avg = brain_df[f"{name}_error"].mean()
        print(f"    {name:15} mean: {brain_avg:.3f}")

    # Methods
    for method in method_names:
        if f"{method}_error" in brain_df.columns:
            brain_avg = brain_df[f"{method}_error"].mean()
            print(f"    {method:15} mean: {brain_avg:.3f}")
