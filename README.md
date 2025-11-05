
# Spatial brain maps

Spatial brain maps is a tool for viewing 3D gene expression in the Mouse brain. by fetching registration data shared via EBRAINS and ISH data shared via the Allen Institute, spatial brain maps reconstructs the 3D patterns of gene expression in a standardised coordinate space. Data can be reconstructed as either a 3D volume, or point cloud. 

## Usage 
While this package is focused on creating 3D volumes, if you wish to search the maps and make region based queries use our public search interface hosted at
https://neural-systems-at-uio.github.io/spatial_brain_maps/. This is integrated with the Siibra explorer atlas viewer so you can interactively search for genes which are up or down regulated in any given region, and explore those volumes of gene expression in 3D. The Python package by contrast allows you to generate new volumes for any given gene at different resolutions, in addition to creating point clouds data. 



```python
from generate_gene_data import gene_to_volume, write_nifti

# 1. Single experiment volume (returns a numpy array)
vol = gene_to_volume('Adora2a', resolution=25)

# 2. Save a volume to NIfTI 
write_nifti(vol, resolution=25, output_path="outputs/exp71717640")
```
This can be used alongside packages such as [brainglobe-atlasapi](https://github.com/brainglobe/brainatlas-api) for plotting and visualisation (see [examples/plot_with_brainglobe.py](examples/plot_with_brainglobe.py)) 

![A horizontal section through the Adora2 gene on top of a Nissl stained reference template.](https://github.com/Neural-Systems-at-UIO/spatial_brain_maps/blob/main/examples/outputs/Adora2a_horizontal.png?raw=true) 

### 2. Producing point clouds. 


## License

Distributed under the terms of the MIT License  (see `LICENSE`).

## Acknowledgements

- Allen Institute for Brain Science for raw ISH data, segmentations, and the Common Coordinate Framework.

- These tools were developed with support from the EBRAINS infrastructure, and funding from the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Framework Partnership Agreement No. 650003 (HBP FPA).

---

Feel free to open issues or pull requests for feature requests, bug reports, or improvements.

