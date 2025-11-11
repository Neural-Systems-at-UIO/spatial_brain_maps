import os
from tqdm import tqdm
import nibabel as nib
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import thin
from skimage.segmentation import find_boundaries
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from matplotlib import colors
from matplotlib.colorbar import ColorbarBase
import matplotlib as mpl  # [ADD] ensure SVG preserves font as text

# ─── user‐tweakable constants ──────────────────────────────────────────────────
VOL_INDEX = 154
OUR_VOXEL_SIZE = 25
TARGET_VOXEL_SIZE = 10
ALLEN_VOXEL_SIZE = 200
NRRD_PAD = ((0, 0), (3, 2), (0, 0))
NRRD_AXES = (0, 2, 1)
GENES_LIST = ["Cap1", "Cacna1g", "Satb1", "Heatr5b"]
OUT_DIR = "datafiles"
FIGURE_DIR = os.path.join("plots")
SHOW_OUTLINES = True  # set to False to disable atlas outlines
# ───────────────────────────────────────────────────────────────────────────────
suffix = "" if SHOW_OUTLINES else "no_"
# ensure output dirs exist
os.makedirs(FIGURE_DIR, exist_ok=True)

# [ADD] preserve fonts as text (don’t convert glyphs to paths) in SVG outputs
mpl.rcParams["svg.fonttype"] = "none"


def load_atlas(name="ccfv3augmented_mouse_10um"):
    atlas = BrainGlobeAtlas(name)
    annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
    outline = (
        find_boundaries(annot, mode="inner", connectivity=annot.ndim)
        if SHOW_OUTLINES
        else None
    )
    return annot, outline, atlas


def load_nifti(path):
    return nib.load(path).get_fdata()


def load_and_prepare_nrrd(path, pad=NRRD_PAD, axes=NRRD_AXES):
    vol, _ = nrrd.read(path)
    vol = np.transpose(vol, axes)[::-1, ::-1, ::-1]
    return np.pad(vol, pad)


def extract_section(vol, index, axis=0):
    if axis == 0:
        return vol[index]
    elif axis == 1:
        return vol[:, index]
    else:
        return vol[:, :, index]


def compute_extent(shape, voxel_size):
    return [0, shape[1] * voxel_size, 0, shape[0] * voxel_size]


def colourbar_save_path(save_path: str) -> str:
    """
    Insert '_colourbar' before the file extension of save_path and force .svg.
    Example: 'plots/foo.png' -> 'plots/foo_colourbar.svg'
    """
    root, _ = os.path.splitext(save_path)
    return f"{root}_colourbar.svg"


def save_colourbar(
    cmap,
    vmin,
    vmax,
    save_path,
    orientation: str = "vertical",
    dpi: int = 300,
    tick_side: str = "right",  # 'left' or 'right' for vertical bars
):
    """
    Save a standalone colourbar figure that matches the heatmap scaling.
    tick_side controls whether ticks are drawn on the left or right for vertical bars.
    """
    fig_size = (0.6, 3.6) if orientation == "vertical" else (3.6, 0.6)
    fig, ax = plt.subplots(figsize=fig_size)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation)

    # Ticks position: left for Allen volumes, right otherwise
    if orientation == "vertical":
        side = "left" if tick_side == "left" else "right"
        cb.ax.yaxis.set_ticks_position(side)
        cb.ax.yaxis.set_label_position(side)
    else:
        side = (
            "bottom" if tick_side == "left" else "top"
        )  # not used here, but supported

    cb.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  → Saving colourbar to {save_path}")
    # Explicitly save as SVG to keep vector text
    fig.savefig(save_path, bbox_inches="tight", format="svg", dpi=dpi)
    plt.close(fig)


def plot_outline(
    section, atlas_sec, sz_vol, sz_atlas, save_path, cmap="magma", vmax_ratio=0.7
):
    # ensure the color limits used for the heatmap are also used for the colourbar
    vmin = float(np.nanmin(section))
    vmax = float(np.nanmax(section) * vmax_ratio)

    ext_vol = compute_extent(section.shape, sz_vol)
    plt.figure(figsize=(12, 10))
    plt.imshow(
        section,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=ext_vol,
        origin="lower",
    )
    if SHOW_OUTLINES and atlas_sec is not None:
        ext_atlas = compute_extent(atlas_sec.shape, sz_atlas)
        plt.contour(
            atlas_sec,
            levels=[0.5],
            colors="white",
            linewidths=0.8,
            extent=ext_atlas,
            alpha=0.8,
        )
    plt.axis("off")
    plt.tight_layout()
    # save main figure (kept as before)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  → Saving figure to {save_path}")
    plt.savefig(save_path)
    plt.close()

    # save matching colourbar image next to the figure (SVG with preserved font)
    cb_path = colourbar_save_path(save_path)
    # Heuristic: files named like 'allen_*.png' are the Allen volumes → ticks on left
    is_allen = os.path.basename(save_path).startswith("allen_")
    tick_side = "left" if is_allen else "right"
    save_colourbar(
        cmap=cmap, vmin=vmin, vmax=vmax, save_path=cb_path, tick_side=tick_side
    )


def save_volume_nrrd(vol, path, spacings):
    header = {"spacings": spacings}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"  → Writing NRRD to {path}")
    nrrd.write(path, vol, header)


# ─── script starts here ────────────────────────────────────────────────────────
print("Loading atlas and computing outline…")
atlas_annot, atlas_outline, atlas = load_atlas()

# 1) NIfTI‐based plots
print("\n1) Generating NIfTI‐based outline plots")
for gene in tqdm(GENES_LIST, desc="NIfTI → outline"):
    print(f"Processing gene: {gene}")
    vol_path = os.path.join(OUT_DIR, "gene_volumes", f"{gene}.nii.gz")
    vol = load_nifti(vol_path)
    sec = extract_section(vol, VOL_INDEX, axis=0)
    atlas_sec = thin(atlas_outline[int(VOL_INDEX * 2.5)]) if SHOW_OUTLINES else None
    save_path = os.path.join(FIGURE_DIR, f"{gene}_atlas_{suffix}outline.png")
    plot_outline(
        sec,
        atlas_sec,
        sz_vol=OUR_VOXEL_SIZE,
        sz_atlas=TARGET_VOXEL_SIZE,
        save_path=save_path,
        vmax_ratio=0.7,
    )

# 2) NRRD‐based plots
print("\n2) Generating NRRD‐based outline plots")
for gene in tqdm(GENES_LIST, desc="NRRD → outline"):
    print(f"Processing gene: {gene}")
    vol_path = os.path.join(OUT_DIR, f"average_allen_{gene}.nrrd")
    vol = load_and_prepare_nrrd(vol_path)
    sec = extract_section(vol, VOL_INDEX // 8, axis=0)
    atlas_sec = thin(atlas_outline[int(VOL_INDEX * 2.5)]) if SHOW_OUTLINES else None
    save_path = os.path.join(FIGURE_DIR, f"allen_{gene}_atlas_{suffix}outline.png")
    plot_outline(
        sec,
        atlas_sec,
        sz_vol=ALLEN_VOXEL_SIZE,
        sz_atlas=TARGET_VOXEL_SIZE,
        save_path=save_path,
        vmax_ratio=1.0,
    )

# 3) Save masks: thalamic areas (ID 549), whole brain, subthalamic nucleus (ID 470), olivary pretectal nucleus (ID 706)
print("\n3) Saving thalamic areas mask (ID 549)")
thal_ids = list(atlas.hierarchy.expand_tree(549))
mask = np.isin(atlas_annot, thal_ids).astype(int) + (atlas_annot != 0).astype(int)
mask_path = os.path.join("datafiles/thalamic_areas.nrrd")
save_volume_nrrd(mask.astype(np.uint8), mask_path, spacings=[0.01, 0.01, 0.01])

print("\n3b) Saving whole-brain mask (atlas_annot != 0)")
brain_mask = (atlas_annot != 0).astype(np.uint8)
brain_mask_path = os.path.join("datafiles/whole_brain_mask.nrrd")
save_volume_nrrd(brain_mask, brain_mask_path, spacings=[0.01, 0.01, 0.01])

print("\n3c) Saving subthalamic nucleus mask (ID 470)")
stn_ids = list(atlas.hierarchy.expand_tree(470))
stn_mask = np.isin(atlas_annot, stn_ids) + (atlas_annot != 0).astype(int)
stn_path = os.path.join("datafiles/subthalamic_nucleus_mask.nrrd")
save_volume_nrrd(stn_mask.astype(np.uint8), stn_path, spacings=[0.01, 0.01, 0.01])

print("\n3d) Saving olivary pretectal nucleus mask (ID 706)")
opn_ids = list(atlas.hierarchy.expand_tree(706))
opn_mask = np.isin(atlas_annot, opn_ids).astype(np.uint8) + (atlas_annot != 0).astype(
    int
)
opn_path = os.path.join("datafiles/olivary_pretectal_nucleus_mask.nrrd")
save_volume_nrrd(opn_mask.astype(np.uint8), opn_path, spacings=[0.01, 0.01, 0.01])

# 4) Export selected genes as NRRD
print("\n4) Exporting Plekhg1, Gal, and Pitx2 volumes to NRRD")

# Plekhg1
plekh = load_nifti(os.path.join(OUT_DIR, "gene_volumes", "Plekhg1.nii.gz"))
plekh_path = os.path.join("datafiles/Plekhg1.nrrd")
save_volume_nrrd(plekh, plekh_path, spacings=[0.025, 0.025, 0.025])

# Gal
gal = load_nifti(os.path.join(OUT_DIR, "gene_volumes", "Gal.nii.gz"))
gal_path = os.path.join("datafiles/Gal.nrrd")
save_volume_nrrd(gal, gal_path, spacings=[0.025, 0.025, 0.025])

# Pitx2
pitx = load_nifti(os.path.join(OUT_DIR, "gene_volumes", "Pitx2.nii.gz"))
pitx_path = os.path.join("datafiles/Pitx2.nrrd")
save_volume_nrrd(pitx, pitx_path, spacings=[0.025, 0.025, 0.025])

# 5) Example coronal slice plot for “Plekhg1”
print("\n5) Plotting coronal slice for Plekhg1")
cor_idx = 262
plekh_cor_sec = extract_section(plekh, cor_idx, axis=1)
plekh_coutline = (
    find_boundaries(atlas_annot[:, int(cor_idx * 2.5)]) if SHOW_OUTLINES else None
)
plekh_cor_path = os.path.join("plots", f"Plekhg1_cor_{suffix}outline.png")
plot_outline(
    plekh_cor_sec,
    plekh_coutline,
    sz_vol=OUR_VOXEL_SIZE,
    sz_atlas=TARGET_VOXEL_SIZE,
    save_path=plekh_cor_path,
)
# Additional: coronal slice with ONLY thalamic areas (ID 549) outline
print("5b) Plotting coronal slice for Plekhg1 with thalamic (ID 549) outline only")
if SHOW_OUTLINES:
    thal_ids = list(atlas.hierarchy.expand_tree(549))
    thal_mask = np.isin(atlas_annot, thal_ids).astype(int)
    plekh_thal_coutline = find_boundaries(thal_mask[:, int(cor_idx * 2.5)])
    plekh_cor_thal_path = os.path.join(
        "plots", f"Plekhg1_cor_thalamus_{suffix}outline.png"
    )
    plot_outline(
        plekh_cor_sec,
        plekh_thal_coutline,
        sz_vol=OUR_VOXEL_SIZE,
        sz_atlas=TARGET_VOXEL_SIZE,
        save_path=plekh_cor_thal_path,
    )

# Additional: whole brain boundary
print("5c) Plotting coronal slice for Plekhg1 with whole-brain boundary")
if SHOW_OUTLINES:
    brain_mask_slice = (atlas_annot != 0).astype(int)
    plekh_brain_coutline = find_boundaries(brain_mask_slice[:, int(cor_idx * 2.5)])
    plekh_cor_brain_path = os.path.join(
        "plots", f"Plekhg1_cor_brain_{suffix}outline.png"
    )
    plot_outline(
        plekh_cor_sec,
        plekh_brain_coutline,
        sz_vol=OUR_VOXEL_SIZE,
        sz_atlas=TARGET_VOXEL_SIZE,
        save_path=plekh_cor_brain_path,
    )

# Additional: both thalamic and whole-brain boundaries overlaid
print("5d) Plotting coronal slice for Plekhg1 with thalamic + whole-brain boundaries")
if SHOW_OUTLINES:
    # Combine boundaries (logical OR) so both are drawn; color remains white in plot_outline
    combined_coutline = (plekh_thal_coutline | plekh_brain_coutline).astype(int)
    plekh_cor_thal_brain_path = os.path.join(
        "plots", f"Plekhg1_cor_thalamus_brain_{suffix}outline.png"
    )
    plot_outline(
        plekh_cor_sec,
        combined_coutline,
        sz_vol=OUR_VOXEL_SIZE,
        sz_atlas=TARGET_VOXEL_SIZE,
        save_path=plekh_cor_thal_brain_path,
    )


# 6) Example coronal slice plots for “Pitx2” (subthalamic nucleus ID tree 470)
print(
    "\n6) Plotting coronal slice for Pitx2 (subthalamic nucleus + full atlas outlines)"
)
pitx = load_nifti(os.path.join(OUT_DIR, "gene_volumes", "Pitx2.nii.gz"))
cidx = cor_idx  # reuse the same coronal index defined earlier (262)
csec = extract_section(pitx, cidx, axis=1)

if SHOW_OUTLINES:
    # Subthalamic nucleus only (ID 470)
    stn_ids = list(atlas.hierarchy.expand_tree(470))
    stn_mask = np.isin(atlas_annot, stn_ids).astype(int)
    pitx_stn_coutline = find_boundaries(stn_mask[:, int(cidx * 2.5)])
    # Full atlas boundaries
    pitx_all_coutline = find_boundaries(atlas_annot[:, int(cidx * 2.5)])
else:
    pitx_stn_coutline = None
    pitx_all_coutline = None

# Plot: subthalamic nucleus + whole brain outline
if SHOW_OUTLINES:
    brain_mask_slice = (atlas_annot != 0).astype(int)
    pitx_brain_coutline = find_boundaries(brain_mask_slice[:, int(cidx * 2.5)])
    pitx_combined_coutline = (pitx_stn_coutline | pitx_brain_coutline).astype(int)
else:
    pitx_combined_coutline = None

pitx_stn_path = os.path.join(
    "plots", f"Pitx2_cor_subthalamic_brain_{suffix}outline.png"
)
plot_outline(
    csec,
    pitx_combined_coutline,
    sz_vol=OUR_VOXEL_SIZE,
    sz_atlas=TARGET_VOXEL_SIZE,
    save_path=pitx_stn_path,
)

# Plot: full atlas boundaries
pitx_all_path = os.path.join("plots", f"Pitx2_cor_atlas_{suffix}outline.png")
plot_outline(
    csec,
    pitx_all_coutline,
    sz_vol=OUR_VOXEL_SIZE,
    sz_atlas=TARGET_VOXEL_SIZE,
    save_path=pitx_all_path,
)

# 6b) Same two plots for “Gal” (olivary pretectal nucleus + whole brain + full atlas outlines)
print(
    "\n6b) Plotting coronal slice for Gal (olivary pretectal nucleus + whole brain + full atlas outlines)"
)
gal = load_nifti(os.path.join(OUT_DIR, "gene_volumes", "Gal.nii.gz"))
cidx = 241
gsec = extract_section(gal, cidx, axis=1)

if SHOW_OUTLINES:
    # Olivary pretectal nucleus (ID 706)
    opn_ids = list(atlas.hierarchy.expand_tree(706))
    opn_mask = np.isin(atlas_annot, opn_ids).astype(int)
    gal_opn_coutline = find_boundaries(opn_mask[:, int(cidx * 2.5)])
    # Whole brain outer boundary
    brain_mask_slice = (atlas_annot != 0).astype(int)
    gal_brain_coutline = find_boundaries(brain_mask_slice[:, int(cidx * 2.5)])
    gal_combined_coutline = (gal_opn_coutline | gal_brain_coutline).astype(int)
    # Full atlas boundaries
    gal_all_coutline = find_boundaries(atlas_annot[:, int(cidx * 2.5)])
else:
    gal_combined_coutline = None
    gal_all_coutline = None

# Plot: olivary pretectal nucleus + whole brain outline
gal_opn_path = os.path.join(
    "plots", f"Gal_cor_olivary_pretectal_brain_{suffix}outline.png"
)
plot_outline(
    gsec,
    gal_combined_coutline,
    sz_vol=OUR_VOXEL_SIZE,
    sz_atlas=TARGET_VOXEL_SIZE,
    save_path=gal_opn_path,
)

# Plot: full atlas boundaries
gal_all_path = os.path.join("plots", f"Gal_cor_atlas_{suffix}outline.png")
plot_outline(
    gsec,
    gal_all_coutline,
    sz_vol=OUR_VOXEL_SIZE,
    sz_atlas=TARGET_VOXEL_SIZE,
    save_path=gal_all_path,
)
