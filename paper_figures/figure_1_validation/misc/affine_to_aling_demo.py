"""This script provides an example of how to apply """


from paper_figures.validation_study.affine_utliities import update_alignment_with_ants_affine,  read_ants_affine
import numpy as np
import cv2
from glob import glob
import json
import copy
from tqdm import tqdm

np.set_printoptions(suppress=True)


root_dir =  r"/home/harryc/github/spatial_brain_maps/paper_figures/validation_study/"




brains = glob(f"{root_dir}/raters/pipeline_registrations/ds_human/*")
for brain in tqdm(brains):
    b = brain.split('/')[-1].split('.')[0]
    print(b)
    with open(brain, 'r') as file:
        data = json.load(file)
        out = copy.deepcopy(data)
        new_slices = []
    for  index,section in enumerate(data['slices'][:]):
        nr = section['nr']
        alignment = np.array(section['anchoring'])
        image_path =glob(f"{root_dir}/section_images/{b}/thumbnails/*s{str(nr).zfill(4)}.jpg")
        affine_path = glob(f"{root_dir}/raters/pipeline_registrations/affine/{b}/*/*_s{str(nr).zfill(4)}_SyN_affineTransfo.mat")
        if image_path:
            image_path = image_path[0]
            affine_path = affine_path[0]
        else:
            continue
        MATRIX = read_ants_affine(affine_path)
        MATRIX = MATRIX[:2,:]
        o_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orig_size = (np.array((o_image.shape[1], o_image.shape[0]) )/ 2.5).astype(int)
        o_image = cv2.resize(o_image, orig_size)
        third = np.percentile(o_image, 2)
        o_image=o_image- third
        ninetyseven = np.percentile(o_image, 95)
        o_image= o_image / ninetyseven
        o_image =o_image * 255
        o_image = o_image.astype(np.uint8)
        final_alignment = update_alignment_with_ants_affine(alignment, MATRIX, o_image.shape)
        section['anchoring'] = list(final_alignment)
        new_slices.append(section)
    out['slices'] = new_slices
    with open(f"{root_dir}/raters/pipeline_registrations/ds_human_affine/{b}.json", 'w') as file:
        json.dump(out, file)
    
    
