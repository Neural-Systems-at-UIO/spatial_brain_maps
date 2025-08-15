from setuptools import setup, find_packages

setup(
    name="spatial_brain_maps",
    version="0.1.0",
    description="Spatial brain maps is a tool for viewing 3D gene expression in the Mouse brain. by fetching registration data shared via EBRAINS and ISH data shared via the Allen Institute, spatial brain maps reconstructs the 3D patterns of gene expression in a standardised coordinate space. Data can be reconstructed as either a 3D volume, or point cloud.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pynutil",
        "opencv-python",
        "numpy",
        "matplotlib",
        "tqdm",
        "pynrrd",
        "scipy",
        "nibabel",
    ],
    entry_points={
        "console_scripts": [
            "spatial_brain_maps=spatial_brain_maps.main:main",
        ],
    },
)
