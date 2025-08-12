"""
This script shows how the 3D volumes of interpolated gene expression were created.
This is mostly useful for replicating the EBRAINs dataset.
"""
import spatial_brain_maps as sbm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
"""
If you would like to locally store the image data as below you can refer to the code at:
https://github.com/PolarBean/allen_download_utilities
"""
image_folder = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/"
"""
The reg data can be sourced from the EBRAINS dataset at:
"""
reg_folder = "/mnt/g/Allen_Realignment_EBRAINS_dataset/"
meta = sbm.utilities.path_utils.metadata
meta = meta[meta['sleep_state'] == 'Nothing']
genes = meta['gene'].unique() 

def _process_gene(gene):
    gene_vol = sbm.gene_to_volume(
        gene,
        reg_folder=reg_folder,
        image_folder=image_folder,
        do_interpolation=True
    )
    sbm.write_nifti(gene_vol, 25, f"outputs/gene_volumes/{gene}")
    return gene

"""
This is a very memory intensive process. Our machine had 256 GB of RAM. 
You may either want to use the non multithreaded loop which is commented 
out below. Or lower the threads to something manageable (a rule of thumb 
is 12.8 GB per thread).
"""
for gene in genes:
    _process_gene(gene)
    
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(_process_gene, gene): gene for gene in genes}
    for future in tqdm(as_completed(futures), total=len(genes)):
        gene = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"Error processing {gene}: {e}")