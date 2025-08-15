from skimage.color import rgb2gray
import os
import numpy as np
import ants

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


def pad_image(moving_image, output_shape, transform_constant=0):
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
    return moving_image


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


def convert_to_warpaffine(MATRIX):
    # 1. Augment the matrix to 3x3 to make it invertible
    scipy_matrix_3x3 = np.vstack([MATRIX, [0, 0, 1]])
    # 2. Invert the matrix. This gives us the true inverse operation.
    M_inv = np.linalg.inv(scipy_matrix_3x3)
    # 3. CRITICAL STEP: Rearrange the inverse matrix for OpenCV's (x, y) convention.
    # This accounts for the (row, col) vs (x, y) difference.
    # This matrix will now correctly map input (x,y) to output (x,y) in a forward direction.
    cv2_matrix = np.array(
        [
            [M_inv[1, 1], M_inv[1, 0], M_inv[1, 2]],
            [M_inv[0, 1], M_inv[0, 0], M_inv[0, 2]],
        ]
    )
    return cv2_matrix


def update_alignment_for_crop(padded_alignment, padded_size, original_size):
    """
    Reverses a padding operation on an OUV alignment vector.

    Args:
        padded_alignment (np.ndarray): The 9-element alignment for the padded image.
        padded_size (tuple): The (height, width) of the padded image.
        original_size (tuple): The (height, width) of the original, un-padded image.

    Returns:
        np.ndarray: The 9-element alignment corresponding to the original image.
    """
    # Unpack dimensions and alignment vectors
    O_pad, U_pad, V_pad = (
        padded_alignment[:3],
        padded_alignment[3:6],
        padded_alignment[6:9],
    )
    H_pad, W_pad = padded_size
    H_orig, W_orig = original_size

    # --- 1. Reverse the scaling of U and V ---
    # Calculate the ratios to scale back to the original size
    width_ratio = W_orig / W_pad
    height_ratio = H_orig / H_pad

    U_orig = U_pad * width_ratio
    V_orig = V_pad * height_ratio

    # --- 2. Reverse the translation of O ---
    # Calculate the padding that was added (assuming symmetrical padding)
    pad_left = (W_pad - W_orig) / 2
    pad_top = (H_pad - H_orig) / 2

    # Calculate the origin shift as a fraction of the PADDED vectors
    # This shifts the origin from the padded canvas's top-left to the original image's top-left
    O_orig = O_pad + (U_pad * (pad_left / W_pad)) + (V_pad * (pad_top / H_pad))

    # --- 3. Combine and return the new alignment ---
    cropped_alignment = np.concatenate([O_orig, U_orig, V_orig])
    return cropped_alignment


def update_alignment_with_ants_affine(alignment, ants_affine_matrix, original_shape):
    """
    Updates an alignment vector based on an ANTS affine transform.

    Args:
        alignment (np.ndarray): The 9-element alignment for the original image.
        ants_affine_matrix (np.ndarray): The 2x3 affine matrix from ANTS.
        original_shape (tuple): The (height, width) of the original, un-padded image.
        output_shape (tuple): The (height, width) of the padded/output image.

    Returns:
        np.ndarray: The updated 9-element alignment for the original image after the transform.
    """
    # 1. Convert ANTs matrix to OpenCV warpAffine format
    cv2_matrix = convert_to_warpaffine(ants_affine_matrix)

    # 2. Invert the OpenCV matrix to map output coordinates back to input
    inv_cv2_matrix = np.linalg.inv(np.vstack((cv2_matrix, [0, 0, 1])))[:2, :]
    output_shape = (
        np.linalg.norm(alignment[6:]).astype(int),
        np.linalg.norm(alignment[3:6]).astype(int),
    )

    # 3. Update alignment for the affine transformation on the padded/output image space
    # This gives the alignment for the *pre-transformed* padded image
    padded_alignment_pre_transform = update_alignment_for_affine(
        alignment, (output_shape[1], output_shape[0]), inv_cv2_matrix
    )

    # 4. Update alignment to account for the padding (crop)
    # This brings the alignment back to the original image's coordinate system
    final_alignment = update_alignment_for_crop(
        padded_alignment_pre_transform, output_shape, original_shape
    )

    return final_alignment
