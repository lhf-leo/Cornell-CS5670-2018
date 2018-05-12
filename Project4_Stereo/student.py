# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    height, width, channel = images[0].shape
    albedo = np.zeros((height, width, channel))
    normals = np.zeros((height, width, 3))
    L = lights.T
    left = np.linalg.inv(L.T.dot(L))
    for i in range(height):
        for j in range(width):
            for c in range(channel):
                I = [img[i,j,c] for img in images]
                G = left.dot(L.T.dot(I))
                k = np.linalg.norm(G)
                if k < 1e-7: k = 0
                else: normals[i][j] += G/k
                albedo[i][j][c] = k
    normals /= channel
    return albedo, normals

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    height, width, _ = points.shape
    projections = np.zeros((height, width, 2))
    projection_matrix = K.dot(Rt)

    for h in range(height):
        for w in range(width):
            p = np.append(points[h, w], 1)
            p = projection_matrix.dot(p)
            projections[h, w, 0] = p[0] / p[2]
            projections[h, w, 1] = p[1] / p[2]

    return projections

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    height, width, channel = image.shape
    normalized = np.zeros((height, width, channel * ncc_size ** 2))
    window_offset = ncc_size // 2

    for i in range(window_offset, height-window_offset):
        for j in range(window_offset, width-window_offset):
            new_vec = []
            for c in range(channel):
                patch = image[i-window_offset:i+window_offset+1, j-window_offset: j+window_offset+1, c]
                mean = np.mean(patch)
                new_vec=np.append(new_vec, [(patch - mean).flatten()])

            l2 = np.linalg.norm(new_vec)
            if l2 < 1e-6:
                normalized[i, j] = np.zeros((new_vec.shape))
            else: 
                normalized[i, j] = new_vec / l2
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    height, width, _ = image1.shape
    ncc = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            ncc[i, j] = np.correlate(image1[i, j], image2[i, j])[0]
    return ncc


def form_poisson_equation_impl(height, width, alpha, normals, depth_weight, depth):
    assert alpha.shape == (height, width)
    assert normals is None or normals.shape == (height, width, 3)
    assert depth is None or depth.shape == (height, width)

    row_ind = []
    col_ind = []
    data_arr = []
    b = []
    if depth_weight is None:
        depth_weight = 1
        
    count = 0
    if normals is not None:
        for i in range(height):
            for j in range(width - 1):
                p1 = alpha[i, j]
                p2 = alpha[i, j + 1]
                if not p1 or not p2: continue

                n1 = normals[i, j, 0]
                n2 = -normals[i, j, 2]
                row_ind.append(count)
                row_ind.append(count)
                col_ind.append(i * width + j)
                col_ind.append(i * width + j + 1)
                data_arr.append(-n2)
                data_arr.append(n2)
                b.append(-n1)
                count += 1

        for i in range(height - 1):
            for j in range(width):
                p1 = alpha[i, j]
                p2 = alpha[i + 1, j]
                if not p1 or not p2: continue

                n1 = -normals[i, j, 1]
                n2 = -normals[i, j, 2]
                row_ind.append(count)
                row_ind.append(count)
                col_ind.append(i * width + j)
                col_ind.append((i + 1) * width + j)
                data_arr.append(-n2)
                data_arr.append(n2)
                b.append(-n1)
                count += 1

    if depth is not None:
        for i in range(height):
            for j in range(width):
                if not alpha[i, j]: continue

                row_ind.append(count)
                col_ind.append(i * width + j)
                data_arr.append(depth_weight)
                b.append(depth_weight * depth[i, j])
                count += 1

    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    data_arr = np.array(data_arr)
    b = np.array(b)

    A = csr_matrix((data_arr, (row_ind, col_ind)), shape=(count, width * height))
 
    return A, b
