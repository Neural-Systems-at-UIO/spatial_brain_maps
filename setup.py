from setuptools import setup, find_packages

setup(
    name="spatial_brain_maps",
    version="0.1.0",
    description="Quantify expression in Allen atlas from slices",
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
