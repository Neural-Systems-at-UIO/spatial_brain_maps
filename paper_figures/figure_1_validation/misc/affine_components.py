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
from utilities import generate_target_slice, plot_image_and_alignment

np.set_printoptions(suppress=True)
# atlas = BrainGlobeAtlas("ccfv3augmented_mouse_25um")
# annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]

# raw_outline3d = find_boundaries(
#                     annot,
#                     mode="inner")


"""
The affine is composed of
Translations
Scalings
Shears
Rotations
Reflections

In this script we work out how to apply each one to the image 
and alignment OUV while keeping them in sync. 
O = the position of the top left of the image in atlas space (XYZ)
U = the position relative to O of the top right of the image in atlas space (XYZ)
V = the position relative to O of the bottom left of the image in atlas space (XYZ)
"""


# scale_x, scale_y = 1.1, 1.1
# shear_x, shear_y = 0.1, 0.1
# t_x, t_y = 10, 20
# MATRIX = np.float32([[scale_x, shear_x, t_x],
#                      [shear_y, scale_y, t_y]])
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


affine_path = "/home/harryc/github/allen_quantifier/paper_figures/validation_study/raters/pipeline_registrations/affine/04-0180/276075/101740229_s0220_SyN_affineTransfo.mat"
MATRIX = read_ants_affine(affine_path)
MATRIX = np.linalg.inv(MATRIX)[:2, :]
image_path = r"/home/harryc/github/allen_quantifier/paper_figures/validation_study/section_images/04-0180/thumbnails/101740229_s0220.jpg"
alignment = np.array(
    [
        -24.277802163804665,
        238.49497780015713,
        301.126115669702,
        451.3549660173083,
        -55.239040314140425,
        103.31357540696513,
        69.93112244075695,
        44.368290926514135,
        -360.28507216865376,
    ]
)
U_val = alignment[3:6] / np.linalg.norm(alignment[3:6])
V_val = alignment[6:] / np.linalg.norm(alignment[6:])


t_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
o_image = cv2.resize(
    t_image,
    (
        np.linalg.norm(alignment[3:6]).astype(int),
        np.linalg.norm(alignment[6:]).astype(int),
    ),
)
plot_image_and_alignment(o_image, alignment, raw_outline3d)


def update_alignment_for_affine(alignment, orig_size, M):
    """
    Update alignment vectors for an affine transformation
    alignment: original 9D alignment vector
    orig_size: (width, height) of original image
    M: 2x3 affine transformation matrix
    Returns: new 9D alignment vector
    """
    W, H = orig_size
    O = alignment[:3]
    U = alignment[3:6]
    V = alignment[6:]

    # Extract linear part and translation
    A = M[:, :2]
    t = M[:, 2]

    # Handle degenerate matrices
    if np.linalg.det(A) < 1e-10:
        A_inv = np.linalg.pinv(A)
    else:
        A_inv = np.linalg.inv(A)

    # Transform corners: new image corners -> original image coordinates
    def transform_point(i, j):
        vec = np.array([i, j]) - t
        return A_inv @ vec

    # Get original coordinates for new image corners
    x0, y0 = transform_point(0, 0)  # TL
    x1, y1 = transform_point(W, 0)  # TR
    x2, y2 = transform_point(0, H)  # BL

    # Compute new atlas coordinates
    TL_atlas = O + (x0 / W) * U + (y0 / H) * V
    TR_atlas = O + (x1 / W) * U + (y1 / H) * V
    BL_atlas = O + (x2 / W) * U + (y2 / H) * V

    # Compute new alignment vectors
    O_new = TL_atlas
    U_new = TR_atlas - TL_atlas
    V_new = BL_atlas - TL_atlas

    return np.concatenate([O_new, U_new, V_new])


# Usage for shearing transformation
orig_size = (o_image.shape[1], o_image.shape[0])
M_shear = np.float32([[1, MATRIX[0, 1], 0], [MATRIX[1, 0], 1, 0]])
new_alignment = update_alignment_for_affine(alignment, orig_size, M_shear)
sheared_image = cv2.warpAffine(o_image, M_shear, orig_size, borderValue=255)
plot_image_and_alignment(sheared_image, new_alignment, raw_outline3d)

# Scaling
M_scale = np.float32([[MATRIX[0, 0], 0, 0], [0, MATRIX[1, 1], 0]])
new_align = update_alignment_for_affine(alignment, orig_size, M_scale)
scaled_img = cv2.warpAffine(o_image, M_scale, orig_size)
plot_image_and_alignment(scaled_img, new_align, raw_outline3d)

# Translation
M_trans = np.float32([[1, 0, MATRIX[0, 2]], [0, 1, MATRIX[1, 2]]])
new_align = update_alignment_for_affine(alignment, orig_size, M_trans)
trans_img = cv2.warpAffine(o_image, M_trans, orig_size)
plot_image_and_alignment(trans_img, new_align, raw_outline3d)

# Combined
M_combined = MATRIX
new_align = update_alignment_for_affine(alignment, orig_size, M_combined)
transformed_img = cv2.warpAffine(o_image, M_combined, orig_size)
plot_image_and_alignment(transformed_img, new_align, raw_outline3d)
