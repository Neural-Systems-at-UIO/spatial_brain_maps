import os
from tqdm import tqdm
import nibabel as nib
import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import thin
from skimage.segmentation import find_boundaries
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

# ─── user‐tweakable constants ──────────────────────────────────────────────────
VOL_INDEX         = 154
OUR_VOXEL_SIZE    = 25
TARGET_VOXEL_SIZE = 10
ALLEN_VOXEL_SIZE  = 200
NRRD_PAD          = ((0,0),(3,2),(0,0))
NRRD_AXES         = (0,2,1)
GENES_LIST        = ["Cap1", "Cacna1g", "Satb1", "Heatr5b"]
OUT_DIR           = "../outputs"
FIGURE_DIR        = os.path.join(OUT_DIR, "figure_elements")
# ───────────────────────────────────────────────────────────────────────────────

# ensure output dirs exist
os.makedirs(FIGURE_DIR, exist_ok=True)

def load_atlas(name="ccfv3augmented_mouse_10um"):
    atlas = BrainGlobeAtlas(name)
    annot = np.transpose(atlas.annotation, (2,0,1))[::-1,::-1,::-1]
    outline = find_boundaries(annot, mode="inner", connectivity=annot.ndim)
    return annot, outline, atlas

def load_nifti(path):
    return nib.load(path).get_fdata()

def load_and_prepare_nrrd(path, pad=NRRD_PAD, axes=NRRD_AXES):
    vol, _ = nrrd.read(path)
    vol = np.transpose(vol, axes)[::-1,::-1,::-1]
    return np.pad(vol, pad)

def extract_section(vol, index, axis=0):
    if axis == 0:
        return vol[index]
    elif axis == 1:
        return vol[:, index]
    else:
        return vol[:, :, index]

def compute_extent(shape, voxel_size):
    return [0, shape[1]*voxel_size, 0, shape[0]*voxel_size]

def plot_outline(section, atlas_sec, sz_vol, sz_atlas, save_path,
                 cmap="magma", vmax_ratio=0.7):
    ext_vol   = compute_extent(section.shape, sz_vol)
    ext_atlas = compute_extent(atlas_sec.shape, sz_atlas)
    plt.figure(figsize=(12,10))
    plt.imshow(section,
               cmap=cmap,
               vmax=section.max()*vmax_ratio,
               extent=ext_vol,
               origin="lower")
    plt.contour(atlas_sec,
                levels=[0.5],
                colors="white",
                linewidths=0.8,
                extent=ext_atlas,
                alpha=0.8)
    plt.axis("off")
    plt.tight_layout()
    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"  → Saving figure to {save_path}")
    plt.savefig(save_path)
    plt.close()

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
    vol      = load_nifti(vol_path)
    sec      = extract_section(vol, VOL_INDEX, axis=0)
    atlas_sec= thin(atlas_outline[int(VOL_INDEX*2.5)])
    save_path = os.path.join(
        FIGURE_DIR, f"{gene}_atlas_outline.png")
    plot_outline(sec, atlas_sec,
                 sz_vol=OUR_VOXEL_SIZE,
                 sz_atlas=TARGET_VOXEL_SIZE,
                 save_path=save_path,
                 vmax_ratio=0.7)

# 2) NRRD‐based plots
print("\n2) Generating NRRD‐based outline plots")
for gene in tqdm(GENES_LIST, desc="NRRD → outline"):
    print(f"Processing gene: {gene}")
    vol_path = os.path.join(OUT_DIR, f"average_allen_{gene}.nrrd")
    vol      = load_and_prepare_nrrd(vol_path)
    sec      = extract_section(vol, VOL_INDEX//8, axis=0)
    atlas_sec= thin(atlas_outline[int(VOL_INDEX*2.5)])
    save_path = os.path.join(
        FIGURE_DIR, f"allen_{gene}_atlas_outline.png")
    plot_outline(sec, atlas_sec,
                 sz_vol=ALLEN_VOXEL_SIZE,
                 sz_atlas=TARGET_VOXEL_SIZE,
                 save_path=save_path,
                 vmax_ratio=1.0)

# 3) Save a mask of thalamic areas (ID 549)
print("\n3) Saving thalamic areas mask (ID 549)")
thal_ids = list(atlas.hierarchy.expand_tree(549))
mask     = (np.isin(atlas_annot, thal_ids).astype(int)
            + (atlas_annot!=0).astype(int))
mask_path = os.path.join(FIGURE_DIR, "thalamic_areas.nrrd")
save_volume_nrrd(mask.astype(np.uint8),
                 mask_path,
                 spacings=[0.01,0.01,0.01])

# 4) Export a single gene as NRRD
print("\n4) Exporting Plekhg1 volume to NRRD")
plekh = load_nifti(os.path.join(
    OUT_DIR, "gene_volumes", "Plekhg1.nii.gz"))
plekh_path = os.path.join(FIGURE_DIR, "Plekhg1.nrrd")
save_volume_nrrd(plekh,
                 plekh_path,
                 spacings=[0.025,0.025,0.025])

# 5) Example horizontal slice plot for “Plekhg1”
print("\n5) Plotting horizontal slice for Plekhg1")
hidx      = 180
hsec      = extract_section(plekh, hidx, axis=2)
houtline  = find_boundaries(atlas_annot[:,:,int(hidx*2.5)])
horiz_path = os.path.join(FIGURE_DIR, "Plekhg1_horiz_outline.png")
plot_outline(hsec, houtline,
             sz_vol=OUR_VOXEL_SIZE,
             sz_atlas=TARGET_VOXEL_SIZE,
             save_path=horiz_path)

# 6) Example coronal slice plot for “Pitx2” (ID tree 470)
print("\n6) Plotting coronal slice for Pitx2")
pitx = load_nifti(os.path.join(
    OUT_DIR, "gene_volumes", "Pitx2.nii.gz"))
cor_ids   = list(atlas.hierarchy.expand_tree(470))
cor_mask  = (np.isin(atlas_annot, cor_ids).astype(int)
             + (atlas_annot!=0).astype(int))
cidx      = 262
csec      = extract_section(pitx, cidx, axis=1)
coutline  = find_boundaries(cor_mask[:,int(cidx*2.5)])
cor_path  = os.path.join(FIGURE_DIR, "Pitx2_cor_outline.png")
plot_outline(csec, coutline,
             sz_vol=OUR_VOXEL_SIZE,
             sz_atlas=TARGET_VOXEL_SIZE,
             save_path=cor_path)