"""
This script shows how the 3D volumes of interpolated gene expression were created.
This is mostly useful for replicating the EBRAINs dataset.
"""

# these lines are for the debugger
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import generate_gene_data as ggd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
If you would like to locally store the image data as below you can refer to the code at:
https://github.com/PolarBean/allen_download_utilities
"""
IMAGE_FOLDER = "/mnt/e/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/"
"""
The reg data can be sourced from the EBRAINS dataset at:
"""
REG_FOLDER = "/mnt/e/Allen_Realignment_EBRAINS_dataset/registration_data"
meta = ggd.utilities.path_utils.metadata
meta = meta[meta["sleep_state"] == "Nothing"]

genes = meta["gene"].unique()
genes = [i for i in genes if i != "Nothing"]
len(genes)


def _process_gene(gene_name):
    gene_vol = ggd.gene_to_volume(
        gene_name,
        reg_folder=REG_FOLDER,
        image_folder=IMAGE_FOLDER,
        do_interpolation=True,
    )
    ggd.write_nifti(
        gene_vol,
        25,
        f"/mnt/e/Allen_Realignment_EBRAINS_dataset/gene_volumes_new/{gene_name}",
    )
    return gene_name


"""
This is a very memory intensive process. Our machine had 256 GB of RAM.
You may either want to use the non multithreaded loop which is commented
out below. Or lower the threads to something manageable (a rule of thumb
is 12.8 GB per thread).
"""
# for gene in tqdm(genes):
#     _process_gene(gene)

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(_process_gene, gene): gene for gene in genes}
    for future in tqdm(as_completed(futures), total=len(genes)):
        gene = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error processing {gene}: {e}")
