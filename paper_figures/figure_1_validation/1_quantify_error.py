import os
import json
import numpy as np
from glob import glob
import nibabel as nib
import math
import cv2
from brainglobe_atlasapi import BrainGlobeAtlas
from skimage.segmentation import find_boundaries
from skimage.morphology import thin
import pandas as pd
import csv
from misc.visualign_deformations import transform_vec, triangulate
import re

class LinInt:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def get(self, x):
        return self.y1 + (self.y2 - self.y1) * (x - self.x1) / (self.x2 - self.x1)


class LinReg:
    def __init__(self):
        self.n = 0
        self.Sx = 0
        self.Sy = 0
        self.Sxx = 0
        self.Sxy = 0

    def add(self, x, y):
        self.n += 1
        self.Sx += x
        self.Sy += y
        self.Sxx += x * x
        self.Sxy += x * y
        if self.n >= 2:
            self.b = (self.n * self.Sxy - self.Sx * self.Sy) / (
                self.n * self.Sxx - self.Sx * self.Sx
            )
            self.a = self.Sy / self.n - self.b * self.Sx / self.n

    def get(self, x):
        return self.a + self.b * x



def normalize(arr, idx):
    l = 0
    for i in range(3):
        l += arr[idx + i] * arr[idx + i]
    l = math.sqrt(l)
    for i in range(3):
        arr[idx + i] /= l
    return l


def orthonormalize(arr):
    normalize(arr, 3)
    dot = 0
    for i in range(3):
        dot += arr[i + 3] * arr[i + 6]
    for i in range(3):
        arr[i + 6] -= arr[i + 3] * dot
    normalize(arr, 6)

def propagate(arr):
    arr = arr.copy()
    for slice in arr:
        if "nr" not in slice:
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))

    arr.sort(key=lambda slice: slice["nr"])

    linregs = [LinReg() for i in range(11)]
    count = 0
    for slice in arr:
        if "anchoring" in slice:
            a = slice["anchoring"]
            for i in range(3):
                a[i] += (a[i + 3] + a[i + 6]) / 2
            a.extend(
                [normalize(a, 3) / slice["width"], normalize(a, 6) / slice["height"]]
            )
            for i in range(len(linregs)):
                linregs[i].add(slice["nr"], a[i])
            count += 1

    if count >= 2:
        l = len(arr)
        if not "anchoring" in arr[0]:
            nr = arr[0]["nr"]
            a = [linreg.get(nr) for linreg in linregs]
            orthonormalize(a)
            arr[0]["anchoring"] = a
            count += 1
        if not "anchoring" in arr[l - 1]:
            nr = arr[l - 1]["nr"]
            a = [linreg.get(nr) for linreg in linregs]
            orthonormalize(a)
            arr[l - 1]["anchoring"] = a
            count += 1

        start = 1
        while count < l:
            while "anchoring" in arr[start]:
                start += 1
            next = start + 1
            while not "anchoring" in arr[next]:
                next += 1
            pnr = arr[start - 1]["nr"]
            nnr = arr[next]["nr"]
            panch = arr[start - 1]["anchoring"]
            nanch = arr[next]["anchoring"]
            linints = [LinInt(pnr, panch[i], nnr, nanch[i]) for i in range(len(panch))]
            for i in range(start, next):
                nr = arr[i]["nr"]
                arr[i]["anchoring"] = [linint.get(nr) for linint in linints]
                count += 1
            start = next + 1

        for slice in arr:
            a = slice["anchoring"]
            orthonormalize(a)
            v = a.pop()
            u = a.pop()
            for i in range(3):
                a[i + 3] *= u * slice["width"]
                a[i + 6] *= v * slice["height"]
                a[i] -= (a[i + 3] + a[i + 6]) / 2
    return arr

def load_quint_json(filename, propagate_missing_values=True):
    """
    Reads a VisuAlign JSON file (.waln or .wwrp) and extracts slice information.
    Slices may include anchoring, grid spacing, and other image metadata.

    Parameters
    ----------
    filename : str
        The path to the VisuAlign JSON file.
    apply_damage_mask : bool
        If True, retains 'grid' data in slices; if False, removes it.

    Returns
    -------
    list
        A list of slice dictionaries containing anchoring and other metadata.
    float or None
        Grid spacing if found, otherwise None.
    """
    with open(filename) as f:
        vafile = json.load(f)
    if filename.endswith(".waln") or filename.endswith("wwrp"):
        slices = vafile["sections"]
        vafile["slices"] = slices
        for slice in slices:
            slice["nr"] = int(re.search(r"_s(\d+)", slice["filename"]).group(1))
            if "ouv" in slice:
                slice["anchoring"] = slice["ouv"]

    else:
        slices = vafile["slices"]
    if (len(slices) > 1) & propagate_missing_values:
        slices = propagate(slices)
    vafile["slices"] = slices
    return vafile


# Resolve all relative paths from this script directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# full path to where your per‐section ANTs affine mats live
AFFINE_FOLDER = os.path.join(
    SCRIPT_DIR, "datafiles", "raters", "pipeline_registrations", "affine"
)
atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
raw_outline3d = find_boundaries(annot, mode="inner", connectivity=annot.ndim)
template = np.transpose(atlas.reference, (2, 0, 1))[::-1, ::-1, ::-1]
template = template / template.max()


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


def mean_reference_xyz(ref_aligns, ref_markers, nx, ny, h, w):
    """Map each reference registration to 3D, then average the 3D positions."""
    if ref_markers is None:
        ref_markers = [None] * len(ref_aligns)
    if len(ref_aligns) != len(ref_markers):
        raise ValueError("ref_aligns and ref_markers must have the same length")
    if not ref_aligns:
        raise ValueError("At least one reference alignment is required")

    grid_x, grid_y = flatten_grid(nx, ny)
    reference_xyz = []
    for alignment, marker_set in zip(ref_aligns, ref_markers):
        px = grid_x.copy()
        py = grid_y.copy()
        if marker_set is not None:
            triangulation = triangulate(w, h, marker_set)
            px = (px / nx) * w
            py = (py / ny) * h
            px, py = transform_vec(triangulation, px, py)
            px = (px / w) * nx
            py = (py / h) * ny

        reference_xyz.append(
            np.column_stack(pix_to_xyz(px, py, ny, nx, alignment))
        )

    mean_xyz = np.mean(reference_xyz, axis=0)
    return mean_xyz[:, 0], mean_xyz[:, 1], mean_xyz[:, 2]


# ─── I/O ───────────────────────────────────────────────────────────────────────
def get_slice_json(path, section_nr):
    data = load_quint_json(path)
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
    nl_im = None
    if mats:
        nl_path = (
            mats[0]
            .replace("_SyN_affineTransfo.mat", "_SyN_nonLinearDf.nii.gz")
            .replace("affine", "nonlin")
        )

        if os.path.exists(nl_path):
            nl_im = nib.load(nl_path).get_fdata()
        else:
            print("no nonlinear found at ", nl_path)
    if nl_im is not None:
        ny, nx = nl_im.shape[0], nl_im.shape[1]
    else:
        # Fallback to section dimensions when no nonlinear field exists.
        ny, nx = int(h), int(w)
    px, py = flatten_grid(nx, ny)
    if test_name == "Our Pipeline" and nl_im is not None:
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
    gtx, gty, gtz = mean_reference_xyz(
        ref_aligns, ref_markers, nx, ny, h, w
    )
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
    Returns {human_name: error vs mean(of the other human raters)}.
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
        ref_m = [markers[n] for n in human_names if n != excluded_name]
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

    # Map every rater's deformed grid into 3D before calculating the consensus.
    gtx, gty, gtz = mean_reference_xyz(
        [aligns[name] for name in human_names],
        [markers.get(name) for name in human_names],
        nx,
        ny,
        H,
        W,
    )

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
# 04-0351 was labeled by the allen as having failed QC and they did not provide alignments for it
# 321-0135, 321-0140 and Pdyn-T2A-CreERT2-258309 were not C56BL/6 and so were not a fair comparison
# as this study was looking at C56BL/6
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
        data = load_quint_json(path)
        sec_sets.append({s["nr"] for s in data["slices"]})
    if not sec_sets:
        print(f"No registration files found for brain {brain}; skipping")
        continue
    section_nrs = sorted(set.intersection(*sec_sets))
    if not section_nrs:
        print(f"No shared sections found for brain {brain}; skipping")
        continue

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
save_consolidated_results(results, "datafiles/consolidated_registration_results.csv")

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
