import spatial_brain_maps as sbm
from brainglobe_atlasapi import BrainGlobeAtlas
import matplotlib.pyplot as plt

# This reorients the brainglobe data into the same orientation as our data
atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
ref = atlas.reference
# normalise reference volume to between 0 and 1
ref = ref / ref.max()
# These paths are optional, if not used sbm will retrieve the data when run
image_folder = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/"
reg_folder = "/mnt/g/Allen_Realignment_EBRAINS_dataset/registration_data/"
gene = "Adora2a"
vol = sbm.gene_to_volume(
    gene, reg_folder=reg_folder, image_folder=image_folder, do_interpolation=True
)
# normalise gene volume to between 0 and 1
vol = vol / vol.max()

# We can then plot using matplotlib
index = 150
data_slice = vol[:, index]
template_slice = ref[:, index]
fig, ax = plt.subplots(figsize=(6, 6))
# Base template in grayscale
ax.imshow(template_slice, cmap="gray")
# Overlay gene data; mask very low values for cleaner view
ax.imshow(data_slice, cmap="magma", alpha=0.5)
plt.tight_layout()
fig.savefig("outputs/Adora2a_horizontal.png")
plt.show()
plt.show()
