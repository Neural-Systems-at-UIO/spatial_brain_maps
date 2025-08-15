from tqdm import tqdm
import json
import nrrd
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from skimage.segmentation import find_boundaries

np.random.seed(52)


# Helper to adjust a colour if necessary
def make_color_unique_random(base_rgb, taken, offset_choices=None, max_tries=100):
    """
    Returns a color similar to base_rgb but not in taken.
    Randomly chooses one or more channels to adjust by a small random offset.
    Uses clamping to avoid underflow/overflow.
    If random trials fail, uses a fallback that tests a range of offsets for each channel.
    """
    if offset_choices is None:
        # Small offsets to keep the new color similar to the original.
        offset_choices = [-20, 20]

    for _ in range(max_tries):
        new_rgb = list(base_rgb)
        # Randomly decide how many channels to change: at least one, up to all three.
        indices_to_change = random.sample(range(3), random.randint(1, 3))
        for idx in indices_to_change:
            offset = random.choice(offset_choices)
            new_rgb[idx] = int(np.clip(new_rgb[idx] + offset, 0, 255))
        new_rgb = tuple(new_rgb)
        if new_rgb != base_rgb and new_rgb not in taken:
            return new_rgb

    # Improved fallback: Try a systematic search over a small range for each channel.
    for r_offset in range(-10, 11):
        for g_offset in range(-10, 11):
            for b_offset in range(-10, 11):
                candidate = (
                    int(np.clip(base_rgb[0] + r_offset, 0, 255)),
                    int(np.clip(base_rgb[1] + g_offset, 0, 255)),
                    int(np.clip(base_rgb[2] + b_offset, 0, 255)),
                )
                if candidate != base_rgb and candidate not in taken:
                    return candidate

    raise ValueError("Unable to find a unique color for {}".format(base_rgb))


# Load the unique colour mapping.
with open("datafiles/allen_unique_rgb.json", "r") as f:
    allen_rgb_dict = json.load(f)


atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
# Get and adjust hemisphere atlas.
hemi_atlas = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
hemi_atlas = hemi_atlas[hemi_atlas.shape[0] // 2 :]

volpath = "datafiles/carea_atlas_55_regions.nrrd"
vol, header = nrrd.read(volpath)

# Build a lookup for colours based on the mode structure id in the atlas.
colour_lookup = {}
taken_colors = set()
for cluster_id in tqdm(np.unique(vol)):
    if cluster_id == 0:
        continue
    mask = vol == cluster_id
    hemi_atlas_ids = hemi_atlas[mask]
    if hemi_atlas_ids.size > 0:
        mode_id = np.bincount(hemi_atlas_ids.flatten()).argmax()
    else:
        mode_id = None
    # Get the base colour from the unique atlas mapping (keys are strings).
    base_color = tuple(allen_rgb_dict.get(str(mode_id), [0, 0, 0]))
    if base_color in taken_colors:
        # Nudge the colour to generate a new unique variant.
        new_color = make_color_unique_random(base_color, taken_colors)
    else:
        new_color = base_color
    colour_lookup[cluster_id] = new_color
    taken_colors.add(new_color)

# Now save the new colour lookup as a .label file for ITK-SNAP.
# ITK-SNAP expects a header and then rows in the format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
output_label_file = "datafiles/careas.label"
with open(output_label_file, "w") as f_label:
    f_label.write("################################################\n")
    f_label.write("# ITK-SnAP Label Description File\n")
    f_label.write("# File format: \n")
    f_label.write("# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n")
    f_label.write("# Fields: \n")
    f_label.write("#    IDX:   Zero-based index\n")
    f_label.write("#    -R-:   Red color component (0..255)\n")
    f_label.write("#    -G-:   Green color component (0..255)\n")
    f_label.write("#    -B-:   Blue color component (0..255)\n")
    f_label.write("#    -A-:   Label transparency (0.00 .. 1.00)\n")
    f_label.write("#    VIS:   Label visibility (0 or 1)\n")
    f_label.write("#    MSH:   Label mesh visibility (0 or 1)\n")
    f_label.write("#    LABEL: Label description\n")
    f_label.write("################################################\n")
    # The first line is the Clear Label.
    f_label.write('0     0    0    0        0  0  0    "Clear Label"\n')
    # Write color entries based on the lookup (clusters).
    # Sorting by cluster_id for consistent order.
    for cluster_id, rgb in sorted(colour_lookup.items()):
        # We use a default transparency=1, visibility=1, mesh visibility=0
        f_label.write(
            f'{cluster_id:<5} {rgb[0]:<5} {rgb[1]:<5} {rgb[2]:<5}    1  1  0    "Carea_{cluster_id}"\n'
        )

# Create a coloured volume based on the lookup.
# The coloured volume will have an extra channel for R, G, B.
colored_vol = np.zeros(vol.shape + (3,), dtype=np.uint8)
for cluster_id, color in colour_lookup.items():
    colored_vol[vol == cluster_id] = np.array(color, dtype=np.uint8)

# Create a new folder for screenshots.
recol_folder = "plots/recolored_carea"
os.makedirs(recol_folder, exist_ok=True)

# Save each slice (along the first axis) of the colored volume as a PNG in the new folder.
num_slices = colored_vol.shape[1]
for i in tqdm(range(num_slices)):
    slice_rec = colored_vol[:, i]
    slice_rec[slice_rec == 0] = 255
    annot_slice = vol[:, i]
    # outline_slice = find_boundaries(annot_slice, connectivity=2, mode='subpixel')
    # outline_mask = outline_slice > 0
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(
        slice_rec, origin="upper", extent=(0, slice_rec.shape[1], slice_rec.shape[0], 0)
    )
    # ax.imshow(
    #     1 - outline_slice,
    #     cmap='gray',
    #     alpha=np.where(outline_mask, 0.7, 0.0),
    #     origin='upper',
    #     extent=(0, slice_rec.shape[1], slice_rec.shape[0], 0)
    # )
    ax.axis("off")
    plt.savefig(
        os.path.join(recol_folder, f"slice_{i:03d}.png"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=300,
    )
    plt.close(fig)
