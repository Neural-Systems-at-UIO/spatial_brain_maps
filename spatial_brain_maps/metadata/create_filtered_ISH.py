from tqdm import tqdm
import pandas as pd
import os

metadata = pd.read_csv("allen_ISH.csv")
path_to_images = "/mnt/g/AllenDataalignmentProj/resolutionPixelSizeMetadata/ISH/"
path_to_registration_files = "/mnt/g/Allen_Realignment_EBRAINS_dataset/"
filtered = []
for i, r in tqdm(metadata.iterrows(), total=len(metadata)):
    path = f"{path_to_registration_files}/affine_registration_files/{r['animal_name']}/{r['experiment_id']}"
    if not os.path.exists(path):
        nissl_path = f"{path_to_registration_files}/affine_registration_files/{r['animal_name']}/NISSL"
        if not os.path.exists(nissl_path):
            continue
    filtered.append(r)
pd.DataFrame(filtered).to_csv("filtered_ISH.csv")
