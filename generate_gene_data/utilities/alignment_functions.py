import ants
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy
import imageio
from skimage.color import rgb2gray
from scipy.ndimage import affine_transform
from .path_utils import pixel_size_from_id
import cv2
from glob import glob
import requests
from io import BytesIO
import os


def read_image(experiment_id, filename, target_resolution, mode, image_folder=None):
    if image_folder is None:
        # --- download straight from Allen API ---
        allen_api = "http://api.brain-map.org/api/v2/image_download/"
        if mode == "expression":
            pix_sz = pixel_size_from_id(experiment_id)
            c = 0
            url = (
                f"{allen_api}{filename}"
                f"?downsample=0&quality=100"
                f"&view=expression&filter=colormap&filterVals=0,1,0,256,0"
            )
        elif mode == "histology":
            pix_sz = 10
            c = 255
            url = f"{allen_api}{filename}?downsample=0&quality=100"
        else:
            raise ValueError("mode must be either expression or histology")

        resp = requests.get(url)
        resp.raise_for_status()
        data = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    else:
        # …existing disk‐load code…
        if mode == "expression":
            img_path = os.path.join(image_folder, "expression", filename + ".jpg")
            pix_sz, c = pixel_size_from_id(experiment_id), 0
        elif mode == "histology":
            img_path = os.path.join(image_folder, "10um_new", filename + ".jpg")
            pix_sz, c = 10, 255
        else:
            raise Exception("mode must be either expression or histology")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    # scale to target_resolution
    scale = pix_sz / target_resolution
    h2, w2 = np.round(np.array(img.shape) * scale).astype(int)
    img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
    return img, c


def load_warped_image(
    filename,
    image_folder,
    affine_folder,
    nonlinear_folder,
    target_resolution,
    mode,
    experiment_id,
):
    affine_path = os.path.join(affine_folder, filename + "_SyN_affineTransfo.mat")
    nonlinear_path = os.path.join(
        nonlinear_folder, filename + "_SyN_nonLinearDF.nii.gz"
    )
    affine_tr = read_ants_affine(affine_path)
    nonlinear_tr, h, w = read_nonlinear(nonlinear_path)

    img, c = read_image(experiment_id, filename, target_resolution, mode, image_folder)

    if (affine_tr is None) or (nonlinear_tr is None):
        return img

    sf = 25 / target_resolution
    affine_tr[[0, 1], [2, 2]] *= sf
    h2, w2 = int(h * sf), int(w * sf)
    nt = cv2.resize(nonlinear_tr * sf, (w2, h2))

    aimg = apply_affine_to_image(
        img, affine_tr, (h2, w2), mode="constant", transform_constant=c
    )
    return apply_nonlinear_to_image(aimg, nt, mode="constant", transform_constant=c)


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


def apply_affine_to_points(affine_matrix, points):
    # Convert the points to homogeneous coordinates
    points_homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    # Apply the transformation
    points_transformed_homogeneous = np.dot(affine_matrix, points_homogeneous.T).T
    # Convert the transformed points back to 2D
    points_transformed_2d = points_transformed_homogeneous[:, :2]
    return points_transformed_2d


def read_nonlinear(non_linear_path):
    if not os.path.exists(non_linear_path):
        return None, None, None
    try:
        non_linear = nib.load(non_linear_path)
        non_linear_data = non_linear.get_fdata()
    except Exception:
        return None, None, None
    # remove dimensions of size 1
    non_linear_data = np.squeeze(non_linear_data)
    height = non_linear_data.shape[0]
    width = non_linear_data.shape[1]
    return non_linear_data, height, width


def apply_nonlinear_to_image(
    moving_image, non_linear_data, mode="nearest", transform_constant=0
):
    non_linear_reorder = np.moveaxis(non_linear_data, [0, 1, 2], [1, 2, 0])
    grid = np.mgrid[0 : moving_image.shape[0], 0 : moving_image.shape[1]]
    warp_grid = non_linear_reorder + grid
    warped_image = scipy.ndimage.map_coordinates(
        moving_image, warp_grid, order=0, mode=mode, cval=transform_constant
    )
    return warped_image


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
    return adjusted_image
