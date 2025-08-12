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
annot = np.transpose(atlas.annotation, (2, 0, 1))[::-1, ::-1, ::-1]
raw_outline3d = find_boundaries(
                    annot,
                    mode="inner")
                    
def generate_target_slice(ouv, atlas):
    """
    Generate a 2D slice from a 3D atlas based on orientation vectors.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        atlas (ndarray): 3D atlas volume.

    Returns:
        ndarray: 2D slice extracted from the atlas.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = ouv
    width = np.floor(math.hypot(ux, uy, uz)).astype(int) + 1
    height = np.floor(math.hypot(vx, vy, vz)).astype(int) + 1
    data = np.zeros((width, height), dtype=np.uint32).flatten()
    xdim, ydim, zdim = atlas.shape

    y_values = np.arange(height)
    x_values = np.arange(width)

    hx = ox + vx * (y_values / height)
    hy = oy + vy * (y_values / height)
    hz = oz + vz * (y_values / height)

    wx = ux * (x_values / width)
    wy = uy * (x_values / width)
    wz = uz * (x_values / width)

    lx = np.floor(hx[:, None] + wx).astype(int)
    ly = np.floor(hy[:, None] + wy).astype(int)
    lz = np.floor(hz[:, None] + wz).astype(int)

    valid_indices = (
        (0 <= lx) & (lx < xdim) & (0 <= ly) & (ly < ydim) & (0 <= lz) & (lz < zdim)
    ).flatten()

    lxf = lx.flatten()
    lyf = ly.flatten()
    lzf = lz.flatten()

    valid_lx = lxf[valid_indices]
    valid_ly = lyf[valid_indices]
    valid_lz = lzf[valid_indices]

    atlas_slice = atlas[valid_lx, valid_ly, valid_lz]
    data[valid_indices] = atlas_slice

    data_im = data.reshape((height, width))
    return data_im

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
    O, U, V      = alignment[:3], alignment[3:6], alignment[6:9]
    offs_u       = xfrac[:, None] * U[None, :]
    offs_v       = yfrac[:, None] * V[None, :]
    pts          = O[None, :] + offs_u + offs_v
    return pts[:, 0], pts[:, 1], pts[:, 2]

def apply_affine_to_points(affine_matrix, points):
    # pre‐shift into padded‐image coords
    pts_h = np.column_stack((points, np.ones(len(points))))
    warped = (affine_matrix @ pts_h.T).T
    return warped[:, :2]
"""code for applying to alignment"""
image_path = r"/home/harryc/github/allen_quantifier/paper_figures/validation_study/section_images/04-0180/thumbnails/101740229_s0220.jpg"
affine_path = "/home/harryc/github/allen_quantifier/paper_figures/validation_study/raters/pipeline_registrations/affine/04-0180/276075/101740229_s0220_SyN_affineTransfo.mat"
affine_matrix = read_ants_affine(affine_path)
alignment = np.array([
                -24.277802163804665,
                214.49497780015713,
                301.126115669702,
                451.3549660173083,
                -55.239040314140425,
                103.31357540696513,
                69.93112244075695,
                44.368290926514135,
                -360.28507216865376
            ])
alignment[1] += 24
align_width = int(np.linalg.norm(alignment[3:6])) + 1
align_height = int(np.linalg.norm(alignment[6:])) + 1
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = np.repeat(image[:,:,np.newaxis], axis=2, repeats=3)
outline = generate_target_slice(alignment, raw_outline3d)
temp_image = cv2.resize(image, (align_width, align_height))
temp_image[outline!=0, :] = 0
plt.imshow(temp_image)

plt.show()

"""code for applying to image"""


def apply_affine_to_image(
    moving_image, affine_matrix, output_shape, mode="constant", transform_constant=0
):
    # convert image to grayscale
    if len(moving_image.shape) == 3:
        moving_image = rgb2gray(moving_image)
    output_height, output_width = output_shape
    pad_top_bottom = output_height - moving_image.shape[0]
    pad_left_right = output_width - moving_image.shape[1]
    pad_top = pad_top_bottom // 2
    pad_bottom = pad_top_bottom - pad_top
    pad_left = pad_left_right // 2
    pad_right = pad_left_right - pad_left
    if pad_top < 0:
        moving_image = moving_image[-pad_top:, :]
        pad_top = 0
    if pad_bottom < 0:
        moving_image = moving_image[:pad_bottom, :]
        pad_bottom = 0
    if pad_left < 0:
        moving_image = moving_image[:, -pad_left:]
        pad_left = 0
    if pad_right < 0:
        moving_image = moving_image[:, :pad_right]
        pad_right = 0

    moving_image = np.pad(
        moving_image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=transform_constant,
    )
    affine_matrix[2, :] = [0, 0, 1]
    adjusted_image = affine_transform(
        moving_image, affine_matrix, order=0, mode="constant", cval=transform_constant
    )
    new_corners = pad_top, pad_bottom, pad_left, pad_right
    return adjusted_image, *new_corners

"""just trying to replicate the crop"""
align_width = int(np.linalg.norm(alignment[3:6])) + 1
align_height = int(np.linalg.norm(alignment[6:])) + 1
temp_image = cv2.resize(image, (int(image.shape[1] // 2.5), int(image.shape[0] // 2.5)))
temp_image, pad_top, pad_bottom, pad_left, pad_right = apply_affine_to_image(temp_image, np.eye(3), (align_height, align_width), transform_constant=0)


align_O = alignment[:3]
align_U = alignment[3:6]
align_V = alignment[6:]
crop_u = (align_U / np.linalg.norm(align_U)) * (pad_left) 
crop_v = (align_V / np.linalg.norm(align_V)) * (pad_top) 
crop_u /= 2
crop_v /= 2
new_O = align_O.copy()
new_U = align_U.copy()
new_V = align_V.copy()
new_O -= (crop_u)
new_O -= (crop_v)
new_U += (crop_u)
new_V += (crop_v)
affed_align = [*new_O, *new_U, *new_V]

outline = generate_target_slice(affed_align, raw_outline3d)
plt.imshow(outline, cmap='gray')
# plt.scatter(aff_pts[:,1], aff_pts[:,0])
# plt.scatter(orig_pts[:,1], orig_pts[:,0])
plt.show()



outline = generate_target_slice(alignment, raw_outline3d)
plt.imshow(outline, cmap='gray')
# plt.scatter(aff_pts[:,1], aff_pts[:,0])
# plt.scatter(orig_pts[:,1], orig_pts[:,0])
plt.show()

align_width = int(np.linalg.norm(alignment[3:6])) + 1
align_height = int(np.linalg.norm(alignment[6:])) + 1
temp_image = cv2.resize(image, (int(image.shape[1] // 2.5), int(image.shape[0] // 2.5)))
temp_image, pad_top, pad_bottom, pad_left, pad_right = apply_affine_to_image(temp_image, affine_matrix, (align_height, align_width), transform_constant=0)
orig_pts = np.array([[pad_top, pad_left], [pad_top, temp_image.shape[1] - pad_right], [temp_image.shape[0] -  pad_bottom, pad_left]])
aff_pts = orig_pts.copy()


aff_pts = apply_affine_to_points(np.linalg.inv(affine_matrix), aff_pts)


outline = generate_target_slice(affed_align, raw_outline3d)
temp_image = cv2.resize(image, (int(np.linalg.norm(affed_align[3:6]))+1, int(np.linalg.norm(affed_align[6:]))+1))

temp_image[outline!=0] = 0
plt.imshow(temp_image, cmap='gray')
plt.scatter(aff_pts[:,1], aff_pts[:,0])
plt.scatter(orig_pts[:,1], orig_pts[:,0])
plt.show()


# # orig_pts = np.array([[0,0], [0,temp_image.shape[1]], [temp_image.shape[0], 0]])
# aff_pts = orig_pts.copy()
# aff_pts = apply_affine_to_points(np.linalg.inv(affine_matrix), aff_pts)
# aff_pts -= orig_pts[0]
# aff_pts_U = aff_pts[:,1] / (orig_pts[1,1]- orig_pts[0,1])
# aff_pts_V = aff_pts[:,0] / (orig_pts[2,0] - orig_pts[0,0])

# align_O = alignment[:3]
# align_U = alignment[3:6]
# align_V = alignment[6:]
# new_O = align_O + (align_U  * aff_pts_U[0]) + (align_V  * aff_pts_V[0]) 
# new_U = (align_U  * aff_pts_U[1]) + (align_V  * aff_pts_V[1]) 
# new_V = (align_U  * aff_pts_U[2]) + (align_V  * aff_pts_V[2]) 



# crop_u = (align_U / np.linalg.norm(align_U)) * pad_left 
# crop_v = (align_V / np.linalg.norm(align_V)) * pad_top 
# crop_u /= 2
# crop_v /= 2
# new_O += crop_u
# new_O += crop_v
# new_U -= crop_u
# new_V -= crop_v
# affed_align = [*new_O, *new_U, *new_V]

# outline = generate_target_slice(affed_align, raw_outline3d)
# temp_image = cv2.resize(image, (int(outline.shape[1]), int(outline.shape[0] )))
# temp_image[outline!=0, :] = 0
# plt.imshow(temp_image)

# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import affine_transform
import cv2

def debug_visualization(image, outline, title, show_padding=True):
    """Enhanced visualization with shape validation"""
    try:
        # Validate shapes
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}")
        if image.shape[:2] != outline.shape:
            print(f"Warning: Shape mismatch - Image: {image.shape}, Outline: {outline.shape}")
            # Resize outline to match image if possible
            outline = cv2.resize(outline.astype(float), (image.shape[1], image.shape[0]))
            outline = (outline > 0.5).astype(np.uint8)
        
        plt.figure(figsize=(12, 6))
        
        # Create overlay
        if len(image.shape) == 2:
            overlay = np.stack([image]*3, axis=-1)
        else:
            overlay = image.copy()
        
        # Safe outline application
        outline_mask = (outline != 0) & (outline.shape[0] == overlay.shape[0]) & (outline.shape[1] == overlay.shape[1])
        overlay[outline_mask] = [255, 0, 0]
        
        if show_padding:
            # Safe padding detection
            if len(image.shape) == 3:
                padding_mask = (image == 255).all(axis=-1)
            else:
                padding_mask = (image == 255)
            overlay[padding_mask] = [0, 0, 255]
        
        plt.imshow(overlay)
        plt.title(f"{title}\nImage: {image.shape}, Outline: {outline.shape}")
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        plt.figure()
        plt.title(f"Error in {title}")
        plt.text(0.1, 0.5, f"Error: {str(e)}", fontsize=12)
        plt.axis('off')
        plt.show()

def safe_pix_to_xyz(px, py, h, w, alignment):
    """Safe version of pix_to_xyz with input validation"""
    try:
        px = np.atleast_1d(px)
        py = np.atleast_1d(py)
        xfrac = px / w
        yfrac = py / h
        O, U, V = alignment[:3], alignment[3:6], alignment[6:9]
        offs_u = xfrac[:, None] * U[None, :]
        offs_v = yfrac[:, None] * V[None, :]
        pts = O[None, :] + offs_u + offs_v
        return pts[:, 0], pts[:, 1], pts[:, 2]
    except Exception as e:
        print(f"Error in pix_to_xyz: {str(e)}")
        raise


# Input validation
if not isinstance(image, np.ndarray):
    raise ValueError("Image must be a numpy array")
if len(alignment) != 9:
    raise ValueError("Alignment vector must have 9 elements")
if affine_matrix.shape != (3, 3):
    raise ValueError("Affine matrix must be 3x3")

# Original and resized dimensions
orig_h, orig_w = image.shape[:2]
resized_h, resized_w = int(orig_h//2.5), int(orig_w//2.5)

# Alignment space dimensions
align_h = int(np.linalg.norm(alignment[6:])) + 1
align_w = int(np.linalg.norm(alignment[3:6])) + 1

# Calculate padding with floor division
pad_h = max(0, (align_h - resized_h))
pad_w = max(0, (align_w - resized_w))
pad_top = pad_h // 2
pad_bottom = pad_h - pad_top
pad_left = pad_w // 2
pad_right = pad_w - pad_left


# Inverse outline transform
resized_img = cv2.resize(image, (resized_w, resized_h))
padded_img = np.pad(
    resized_img,
    ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)),
    mode='constant',
    constant_values=0
)

outline = generate_target_slice(alignment, raw_outline3d)
inv_affine = np.linalg.inv(affine_matrix)
warped_outline = affine_transform(
    outline.astype(float),
    inv_affine,
    output_shape=(align_h, align_w),
    order=0
    )
padded_img[warped_outline!=0,:] = 0
plt.imshow(padded_img)
plt.scatter(new_points[:,0],new_points[:,1])
plt.scatter(points[:,0],points[:,1])

plt.show()