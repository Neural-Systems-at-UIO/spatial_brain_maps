import concurrent.futures

# full k means
from sklearn.cluster import KMeans
import time
import nrrd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from tqdm import tqdm

# parameters
cluster_list = [55]
files = glob("/mnt/e/Allen_Realignment_EBRAINS_dataset/CArea_atlas/pca_new/*.nrrd")

atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
hemi_atlas = atlas.annotation
hemi_atlas = (hemi_atlas[:,:,: hemi_atlas.shape[2] // 2][:,:,::-1] )
hemimask = hemi_atlas != 0

def read_nrrd(file):
    data, _ = nrrd.read(file)
    data = data[hemimask]
    return data


# Multithreaded loading of volumes with working progress bar
volumes = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit all tasks first
    futures = [executor.submit(read_nrrd, file) for file in files]

    # Use explicit tqdm updating to ensure the bar increments as futures finish.
    with tqdm(total=len(futures), desc="Loading volumes") as pbar:
        for f in concurrent.futures.as_completed(futures):
            volumes.append(f.result())
            pbar.update(1)

# Stack into a single numpy array
volumes = np.stack(volumes, axis=0)
for n_clusters in cluster_list:
    # Use fewer initializations for large k
    if n_clusters == 32768:
        curr_n_init = 1
        cur_max_iter = 500
    elif n_clusters >= 1000:
        curr_n_init = 5
        cur_max_iter = 1000
    else:
        curr_n_init = 200
        cur_max_iter = 10000

    print(f"\nRunning full KMeans with k={n_clusters}, n_init={curr_n_init}")
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=cur_max_iter,
        n_init=curr_n_init,
        random_state=42,
        verbose=0,
    )
    start = time.time()
    km.fit(volumes.T)
    duration = time.time() - start
    inertia = km.inertia_
    labels = km.labels_
    # save NRRD
    output = np.zeros_like(hemi_atlas)
    output[hemimask] = labels + 1
    output = output.astype(np.uint8)
    nrrd.write(
        f"/mnt/e/outputs/clusters/new_init_{curr_n_init}_full_test_auto_{n_clusters}_regions.nrrd",
        output,
    )
