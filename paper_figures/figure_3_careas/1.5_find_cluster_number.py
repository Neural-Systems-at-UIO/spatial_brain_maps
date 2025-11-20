import concurrent.futures
import time
from glob import glob

import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
from pathlib import Path
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


# parameters
cluster_list = np.arange(10, 100, 5)
sample_size = 1_000_000
random_seed = 42
max_iter = 1000
files = glob("/mnt/e/Allen_Realignment_EBRAINS_dataset/CArea_atlas/pca_new/*.nrrd")
if len(files) == 0:
    raise FileNotFoundError("No PCA volumes found")


# load atlas and hemisphere mask (same as original but reused for subsampling)

atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")

hemi_atlas = atlas.annotation
hemi_atlas = (hemi_atlas[:,:,: hemi_atlas.shape[2] // 2][:,:,::-1] / 2) + (
    hemi_atlas[:,:,hemi_atlas.shape[2] // 2 :] / 2
)
hemimask = hemi_atlas != 0

n_voxels = hemimask.sum()

if sample_size > n_voxels:
    raise ValueError("sample_size larger than available hemisphere voxels")

rng = np.random.default_rng(random_seed)
sample_indices = np.sort(rng.choice(n_voxels, size=sample_size, replace=False))


def read_sampled_nrrd(path):
    data, _ = nrrd.read(path)
    data = data[hemimask]
    return data[sample_indices].astype(np.float32)


def estimate_elbow(xs, ys, *, increasing, curve, interp_points=500):
    """Return (k, metric_value) at the estimated elbow or None using kneed."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 3:
        return None
    xs = xs[mask]
    ys = ys[mask]
    if not np.all(np.diff(xs) >= 0):
        sort_idx = np.argsort(xs)
        xs = xs[sort_idx]
        ys = ys[sort_idx]
    interp_points = max(int(interp_points), xs.size)
    if interp_points <= xs.size:
        x_dense = xs
        y_dense = ys
    else:
        x_dense = np.linspace(xs.min(), xs.max(), num=interp_points)
        y_dense = np.interp(x_dense, xs, ys)
    direction = "increasing" if increasing else "decreasing"
    try:
        kneedle = KneeLocator(
            x_dense,
            y_dense,
            curve=curve,
            direction=direction,
            S=1.0,
            online=False,
            interp_method="polynomial",
        )
    except (RuntimeError, ValueError):
        kneedle = None
    elbow = kneedle.elbow if kneedle is not None else None
    if elbow is not None:
        elbow_y = getattr(kneedle, "elbow_y", None)
        if elbow_y is None or not np.isfinite(elbow_y):
            elbow_y = np.interp(elbow, x_dense, y_dense)
        return float(elbow), float(elbow_y)
    return None


# Multithreaded loading of sampled volumes with progress bar
volumes = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(read_sampled_nrrd, file) for file in files]
    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        volumes.append(f.result())

# Stack into a single numpy array (shape n_volumes x sample_size)
volumes = np.stack(volumes, axis=0)
voxel_matrix = volumes.T  # sample_size x n_volumes
total_ss = float(np.sum((voxel_matrix - voxel_matrix.mean(axis=0, keepdims=True)) ** 2))

# store metrics for plotting
explained_vars = []
silhouette_scores = []
durations = []
ks = []

for n_clusters in tqdm(cluster_list):
    if n_clusters >= voxel_matrix.shape[0]:
        print(f"Skipping k={n_clusters}: more clusters than sampled voxels")
        continue

    # Use fewer initializations for large k (same idea as original script)
    if n_clusters >= 5000:
        curr_n_init = 1
        cur_max_iter = 400
    elif n_clusters >= 1000:
        curr_n_init = 5
        cur_max_iter = max_iter
    elif n_clusters >= 256:
        curr_n_init = 10
        cur_max_iter = max_iter
    else:
        curr_n_init = 50
        cur_max_iter = max_iter

    print(f"\nRunning KMeans with k={n_clusters}, n_init={curr_n_init}")
    km = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=cur_max_iter,
        n_init=curr_n_init,
        random_state=random_seed,
        verbose=0,
    )

    start = time.time()
    km.fit(voxel_matrix)
    duration = time.time() - start
    durations.append(duration)

    labels = km.labels_
    inertia = float(km.inertia_)
    explained_var = 1.0 - inertia / total_ss if total_ss else float("nan")

    ks.append(n_clusters)
    explained_vars.append(explained_var)

    if n_clusters > 1:
        try:
            sil = float(
                silhouette_score(
                    voxel_matrix,
                    labels,
                    metric="euclidean",
                    sample_size=min(5000, voxel_matrix.shape[0]),
                    random_state=random_seed,
                )
            )
        except ValueError:
            sil = float("nan")
    else:
        sil = float("nan")

    silhouette_scores.append(sil)

    print(
        "  metrics: "
        f"inertia={inertia:.3e}, explained_var={explained_var:.4f}, "
        f"silhouette={sil:.4f}, "
        f"duration={duration:.1f}s"
    )


# Plot the metrics so we can eyeball an elbow / good k
ks = np.asarray(ks)
explained_vars = np.asarray(explained_vars)
silhouette_scores = np.asarray(silhouette_scores)

if ks.size:
    sort_idx = np.argsort(ks)
    ks = ks[sort_idx]
    explained_vars = explained_vars[sort_idx]
    silhouette_scores = silhouette_scores[sort_idx]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ks, silhouette_scores, marker="o")
ax.set_ylabel("Silhouette")
ax.set_xlabel("Number of clusters (k)")
ax.grid(True, linestyle="--", alpha=0.3)

# Manual annotation at k=55
manual_k = 55
if manual_k in ks:
    idx = np.where(ks == manual_k)[0][0]
    manual_score = silhouette_scores[idx]

    ax.axvline(manual_k, color="tab:red", linestyle="--", alpha=0.6)
    ax.annotate(
        f"selected k={manual_k}",
        xy=(manual_k, manual_score),
        xytext=(12, 24),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5),
        color="tab:red",
        fontsize=9,
    )
    print(f"Manually selected k={manual_k}")

plt.title("KMeans Silhouette Score vs cluster number")
plt.tight_layout()

# Save to SVG
out_dir = Path("plots")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "kmeans_metrics.svg"
plt.savefig(out_path, format="svg", bbox_inches="tight")
print(f"Saved figure to {out_path}")

plt.show()
