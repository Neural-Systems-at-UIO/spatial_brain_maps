import os
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import nrrd
from tqdm import tqdm
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
import nibabel as nib
from glob import glob
from sklearn.decomposition import IncrementalPCA
from sklearn.utils import gen_batches



atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")

hemi_atlas = np.transpose(atlas.annotation, (2, 0, 1))[::-1,::-1,::-1]
hemi_atlas = ((hemi_atlas[: hemi_atlas.shape[0] // 2][::-1]/ 2) + (hemi_atlas[hemi_atlas.shape[0] // 2 :]/ 2)) 
hemimask = hemi_atlas!=0

hemi_template = np.transpose(atlas.reference, (2, 0, 1))[::-1,::-1,::-1]
hemi_template = ((hemi_template[: hemi_template.shape[0] // 2][::-1]/ 2) + (hemi_template[hemi_template.shape[0] // 2 :]/ 2)) 
hemi_template = hemi_template / hemi_template.max()
hemi_template = hemi_template * 255
hemi_template = hemi_template.astype(np.uint8)

files = glob('outputs/gene_volumes/*.nii.gz')
files = [i for i in files if os.path.exists(i)]
def process_file(file):
    img = nib.load(file)
    arr = img.get_fdata()
    half = arr.shape[0] // 2
    # mirror and average halves
    mirror = arr[:half][::-1] / 2.0
    mirror += arr[half:] / 2.0
    mirror = mirror.astype(np.uint8)
    return mirror[hemimask != 0]



# Multithreaded processing while preserving order
voxels = [None] * (len(files) + 1)
with ThreadPoolExecutor() as executor:
    # submit and keep track of each file's index
    future_to_idx = {
        executor.submit(process_file, f): idx
        for idx, f in enumerate(files)
    }

    # collect results as they complete, but place them at the right index
    for fut in tqdm(as_completed(future_to_idx), total=len(files), desc="Processing files"):
        idx = future_to_idx[fut]
        voxels[idx] = fut.result()

voxels[-1] = hemi_template[hemimask!=0]
 
voxels = np.array(voxels, dtype=np.uint8)
voxels = voxels.T
min_val = voxels.min(axis=0)
max_val = voxels.max(axis=0)
print("min_val_shape: ", min_val.shape)
# Rescale in batches
batch_size = 100_000  # Adjust based on RAM

for i in tqdm(range(0, len(voxels), batch_size)):
    batch = voxels[i:i + batch_size]
    batch_scaled = ((batch - min_val) * (255 / (max_val - min_val))).astype(np.uint8)
    voxels[i:i + batch_size] = batch_scaled


# Settings
n_samples, n_features = voxels.shape
batch_size = 250_000  # Adjust based on your RAM

# Initialize IncrementalPCA
ipca = IncrementalPCA(n_components=None, batch_size=batch_size)

perm = np.random.permutation(n_samples)
for batch in tqdm(gen_batches(n_samples, batch_size), 
                 total=n_samples//batch_size,
                 desc="Fitting PCA"):
    batch_idx = perm[batch]
    batch_data = voxels[batch_idx]
    ipca.partial_fit(batch_data)
 
# Determine number of components to keep
explained_variance_ratio = ipca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
# Find number of components that explain 95% of variance
percent = 0.70
n_components_ = np.argmax(cumulative_variance_ratio >= percent) + 1
print(f"Number of components explaining {percent  * 100}% of variance: {n_components_}")
# Batch transform to avoid memory issues
voxels_pca = np.empty((voxels.shape[0], n_components_), dtype=np.float32)
for i, batch in enumerate(tqdm(gen_batches(voxels.shape[0], batch_size), total=voxels.shape[0] // batch_size, desc="PCA transform")):
    batch_data = voxels[batch]
    voxels_pca[batch, :] = ipca.transform(batch_data)[:, :n_components_]


for i in tqdm(range(n_components_)):
    temp_vol = np.zeros(hemimask.shape)
    temp_vol[hemimask!=0] = voxels_pca[:,i]
    nrrd.write(f"/mnt/g/outputs/pca/pca{i}.nrrd", temp_vol)



