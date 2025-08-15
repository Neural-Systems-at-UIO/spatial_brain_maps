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
cluster_list = [1000, 2000, 4096, 32768]
max_iter = 1000       # maximum number of k-means iterations
n_init = 50           # how many times to run k-means with different centroid seeds
sil_sample = 10000    # for large datasets, subsample for silhouette
files = glob(f"/mnt/g/outputs/pca/*.nrrd")

atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
hemi_atlas = np.transpose(atlas.annotation, (2, 0, 1))[::-1,::-1,::-1]
hemi_atlas = hemi_atlas[hemi_atlas.shape[0] // 2 :]
hemimask = hemi_atlas!=0

def read_nrrd(file):
    data, _ = nrrd.read(file)
    data = data[hemimask]
    return data

# Multithreaded loading of volumes with working progress bar
volumes = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit all tasks first
    futures = [executor.submit(read_nrrd, file) for file in files]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        volumes.append(f.result())

# Stack into a single numpy array
volumes = np.stack(volumes, axis=0)
for n_clusters in cluster_list:
    print(f"\nRunning full KMeans with k={n_clusters}")
    km = KMeans(n_clusters=n_clusters,
                init='k-means++',
                max_iter=max_iter,
                n_init=n_init,
                random_state=42,
                verbose=0)
    start = time.time()
    km.fit(volumes.T)
    duration = time.time() - start
    inertia = km.inertia_
    labels = km.labels_
    # save NRRD
    output = np.zeros_like(hemi_atlas)
    output[hemimask] = labels + 1
    nrrd.write(f'/mnt/g/outputs/clusters/init_{n_init}_full_test_auto_{n_clusters}_regions.nrrd', output)
    