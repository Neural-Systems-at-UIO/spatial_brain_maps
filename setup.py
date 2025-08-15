from setuptools import setup, find_packages
from pathlib import Path

# Automatic versioning is handled by setuptools-scm using git tags.
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="spatial_brain_maps",
    use_scm_version=True,  # derives version from the latest git tag
    license="MIT",
    description=(
        "Spatial brain maps is a tool for viewing 3D gene expression in the Mouse brain. "
        "By fetching registration data shared via EBRAINS and ISH data shared via the Allen Institute, "
        "spatial brain maps reconstructs the 3D patterns of gene expression in a standardised coordinate space. "
        "Data can be reconstructed as either a 3D volume, or point cloud."
    ),
    packages=find_packages(),
    url="https://github.com/Neural-Systems-at-UIO/spatial_brain_maps",
    author="Neural Systems @ UiO",
    author_email="harry.carey@medisin.uio.no",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
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
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
