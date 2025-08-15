from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import itertools
from glob import glob
import nibabel as nib
from scipy.ndimage import map_coordinates
import PyNutil
import ants
import math
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
from scipy.ndimage import affine_transform
from skimage.morphology import thin

atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
# annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
# raw_outline3d = find_boundaries(
#                     annot,
#                     mode="inner")


def calculate_affine(srcPoints, dstPoints):
    # Add a fourth coordinate of 1 to each point
    srcPoints = np.hstack((srcPoints, np.ones((srcPoints.shape[0], 1))))
    dstPoints = np.hstack((dstPoints, np.ones((dstPoints.shape[0], 1))))
    # Solve the system of linear equations
    affine_matrix, _, _, _ = np.linalg.lstsq(srcPoints, dstPoints, rcond=None)
    return affine_matrix.T


def read_ants_affine(aff_path):
    if not os.path.exists(aff_path):
        return None
    ants_affine = ants.read_transform(aff_path)
    before_points = np.array([[0, 0], [0, 1], [1, 0]])
    after_points = np.array([ants_affine.apply_to_point(p) for p in before_points])
    # calculate the affine matrix
    affine_matrix = calculate_affine(before_points, after_points)
    return affine_matrix


def pix_to_xyz(px, py, h, w, alignment):
    xfrac, yfrac = px / w, py / h
    O, U, V = alignment[:3], alignment[3:6], alignment[6:9]
    offs_u = xfrac[:, None] * U[None, :]
    offs_v = yfrac[:, None] * V[None, :]
    pts = O[None, :] + offs_u + offs_v
    return pts[:, 0], pts[:, 1], pts[:, 2]


def apply_affine_to_points(affine_matrix, points):
    # pre‐shift into padded‐image coords
    pts_h = np.column_stack((points, np.ones(len(points))))
    warped = (affine_matrix @ pts_h.T).T
    return warped[:, :2]


"""code for applying to alignment"""
image_path = r"/home/harryc/github/spatial_brain_maps/paper_figures/validation_study/section_images/04-0180/thumbnails/101740229_s0220.jpg"
affine_path = "/home/harryc/github/allen_quantifier/paper_figures/validation_study/raters/pipeline_registrations/affine/04-0180/276075/101740229_s0220_SyN_affineTransfo.mat"
affine_matrix = read_ants_affine(affine_path)
alignment = np.array(
    [
        -24.277802163804665,
        214.49497780015713,
        301.126115669702,
        451.3549660173083,
        -55.239040314140425,
        103.31357540696513,
        69.93112244075695,
        44.368290926514135,
        -360.28507216865376,
    ]
)
alignment[1] += 24
align_width = int(np.linalg.norm(alignment[3:6])) + 1
align_height = int(np.linalg.norm(alignment[6:])) + 1
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5)))

outline = generate_target_slice(alignment, raw_outline3d)
temp_image = cv2.resize(image, (outline.shape[1], outline.shape[0]))

temp_image[outline != 0] = 0
plt.imshow(temp_image)

plt.show()


output_shape = align_height, align_width
temp_image = image.copy()
if len(temp_image.shape) == 3:
    temp_image = rgb2gray(temp_image)
output_height, output_width = output_shape
pad_top_bottom = output_height - temp_image.shape[0]
pad_left_right = output_width - temp_image.shape[1]

pad_top = pad_top_bottom // 2
pad_bottom = pad_top_bottom - pad_top
pad_left = pad_left_right // 2
pad_right = pad_left_right - pad_left
pad_right = pad_right * 10
transform_constant = 0
if pad_top < 0:
    temp_image = temp_image[-pad_top:, :]
    pad_top = 0
if pad_bottom < 0:
    temp_image = temp_image[:pad_bottom, :]
    pad_bottom = 0
if pad_left < 0:
    temp_image = temp_image[:, -pad_left:]
    pad_left = 0
if pad_right < 0:
    temp_image = temp_image[:, :pad_right]
    pad_right = 0

O = alignment[:3]
U = alignment[3:6]
V = alignment[6:]

top_percent = pad_top / temp_image.shape[0]
bottom_percent = pad_bottom / temp_image.shape[0]
left_percent = pad_left / temp_image.shape[1]

right_percent = pad_right / temp_image.shape[1]
O -= U * left_percent
O -= V * top_percent
U += U * (right_percent + left_percent)
V += V * (bottom_percent + top_percent)
pad_alignment = [*O, *U, *V]
temp_image = np.pad(
    temp_image,
    ((pad_top, pad_bottom), (pad_left, pad_right)),
    mode="constant",
    constant_values=transform_constant,
)
outline = generate_target_slice(pad_alignment, raw_outline3d)
temp_image = cv2.resize(temp_image, (outline.shape[1], outline.shape[0]))
temp_image[outline != 0] = 0
plt.imshow(temp_image)
plt.show()
