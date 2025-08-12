from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from skimage.segmentation import find_boundaries
import numpy as np
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import json
from skimage.morphology import thin
from skimage.transform import rescale
from skimage.transform import resize

np.random.seed(42)
atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
annotation = atlas.annotation  # original label volume
# Uncomment or adjust the following line if you require a specific orientation.
annotation = np.transpose(annotation, (2, 0, 1))[::-1,::-1,::-1]
# outline = find_boundaries(annotation, mode="subpixel", connectivity=annotation.ndim)

# Build the original rgb_lookup mapping using each structure's id and its rgb_triplet.
rgb_lookup = {i['id']: i['rgb_triplet'] for _, i in atlas.structures.items()}

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
                    int(np.clip(base_rgb[2] + b_offset, 0, 255))
                )
                if candidate != base_rgb and candidate not in taken:
                    return candidate

    raise ValueError("Unable to find a unique color for {}".format(base_rgb))

# Create a recolored lookup ensuring unique rgb codes.
unique_rgb_lookup = {}
taken_colors = set()

for struct_id, rgb in tqdm(rgb_lookup.items()):
    rgb_tuple = tuple(rgb)  # ensure hashability
    if rgb_tuple in taken_colors:
        # Generate a new unique color by random modifications on some channels.
        rgb_tuple = make_color_unique_random(rgb_tuple, taken_colors)
    unique_rgb_lookup[struct_id] = rgb_tuple
    taken_colors.add(rgb_tuple)

# Create two new RGB volumes.
# The shape is the original annotation shape plus a channel dimension.
volume_shape = annotation.shape + (3,)
og_rgb_volume = np.zeros(volume_shape, dtype=np.uint8)
recoloured_rgb_volume = np.zeros(volume_shape, dtype=np.uint8)

# Fill in the volumes by mapping each structure id to its corresponding rgb triplet.
# For the original RGB volume.
max_label = annotation.max()  # assumes structure ids are non-negative integers
rgb_array = np.zeros((max_label+1, 3), dtype=np.uint8)
unique_rgb_array = np.zeros((max_label+1, 3), dtype=np.uint8)
with open("allen_unique_rgb.json", "w") as f:
    json.dump(unique_rgb_lookup, f)
for struct_id, color in tqdm(rgb_lookup.items()):
    rgb_array[struct_id] = np.array(color, dtype=np.uint8)

for struct_id, color in tqdm(unique_rgb_lookup.items()):
    unique_rgb_array[struct_id] = np.array(color, dtype=np.uint8)

# Now map the annotation directly to an RGB volume.
og_rgb_volume = rgb_array[annotation]
recoloured_rgb_volume = unique_rgb_array[annotation]
# At this point, og_rgb_volume and recoloured_rgb_volume hold your two new volumes.
# You can now save or visualize these volumes as required.

# Directories where the screenshots will be saved.
og_folder = "screenshots/original"
recol_folder = "screenshots/recolored"
os.makedirs(og_folder, exist_ok=True)
os.makedirs(recol_folder, exist_ok=True)
import concurrent.futures

def save_slice(i):
    # Extract the i-th slice for both volumes. 
    # slice_og = og_rgb_volume[:, i]
    slice_rec = recoloured_rgb_volume[:, i]
    annot_slice = annotation[:, i]
    # outline_slice = find_boundaries(annot_slice, connectivity=2, mode='subpixel')
    # Set background to white
    # slice_og = slice_og.copy()
    # slice_og[slice_og == 0] = 255
    slice_rec[slice_rec == 0] = 255

    # Make outline white voxels (==0) transparent
    # outline_mask = outline_slice > 0

    # # Plot original RGB with outline overlay, corners aligned
    # fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    # ax.imshow(slice_og, origin='upper', extent=(0, slice_og.shape[1], slice_og.shape[0], 0))
    # ax.imshow(
    #     outline_slice,
    #     cmap='gray',
    #     alpha=np.where(outline_mask, 0.7, 0.0),
    #     origin='upper',
    #     extent=(0, slice_og.shape[1], slice_og.shape[0], 0)
    # )
    # ax.axis('off')
    # plt.savefig(os.path.join(og_folder, f"slice_{i:03d}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.close(fig)

    # Plot recolored RGB with outline overlay, corners aligned
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.imshow(slice_rec, origin='upper', extent=(0, slice_rec.shape[1], slice_rec.shape[0], 0))
    # ax.imshow(
    #     1 - outline_slice,
    #     cmap='gray',
    #     alpha=np.where(outline_mask, 0.7, 0.0),
    #     origin='upper',
    #     extent=(0, slice_rec.shape[1], slice_rec.shape[0], 0)
    # )
    ax.axis('off')
    plt.savefig(os.path.join(recol_folder, f"slice_{i:03d}.png"), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

# Number of slices along the first dimension.
num_slices = og_rgb_volume.shape[1]
for i in tqdm(range(num_slices)):
    save_slice(i)
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     list(tqdm(executor.map(save_slice, range(num_slices)), total=num_slices))
