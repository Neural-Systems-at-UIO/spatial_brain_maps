import numpy as np
import os
import json
import nibabel as nib
from paper_figures.figure_1_validation.misc.affine_utliities import update_alignment_with_ants_affine,  read_ants_affine
import copy
from glob import glob
from tqdm import tqdm
registration_jsons = glob('/mnt/g/Allen_Realignment_EBRAINS_dataset/registration_data/QUINT_registration_jsons/*.json')
affine_registration_path = "/mnt/g/Allen_Realignment_EBRAINS_dataset/registration_data/affine_registration_files/{}/{}/{}_SyN_affineTransfo.mat"
out_folder = "/mnt/g/Allen_Realignment_EBRAINS_dataset/registration_data/combined_registration_jsons/"
for file in tqdm(registration_jsons):
    brain = file.split('/')[-1].split('.')[0]
    with open(file, 'r') as f:
        data = json.load(f)
        out = copy.deepcopy(data)
        new_slices = []
    for section in data['slices']:
        exp, fn = section['filename'].split('/')
        fn, _ = fn.split('.')
        h = section['height'] // 2.5
        w = section['width'] // 2.5
        affine_path = affine_registration_path.format(brain, exp, fn)
        nr = section['nr']
        alignment = np.array(section['anchoring'])
        if os.path.exists(affine_path):
            MATRIX = read_ants_affine(affine_path)
            MATRIX = MATRIX[:2,:]
            final_alignment = update_alignment_with_ants_affine(alignment, MATRIX, (h,w))
        else:
            final_alignment = alignment
        section['anchoring'] = list(final_alignment)
        new_slices.append(section)
    out['slices'] = new_slices
    with open(f"{out_folder}/{brain}.json", 'w') as file:
        json.dump(out, file)
    