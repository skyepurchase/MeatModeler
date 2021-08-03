import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def project(points, translation_scalars, frame_extrinsic_matrices, frame_indices, camera_matrix):
    """
    Takes an array of 3D points and corresponding camera parameters and returns the re-projected 2D points

    :param points: Array of 3D points
    :param translation_scalars: Array of floats to scale the translation of each extrinsic matrix
    :param frame_extrinsic_matrices: Array of 4x4 extrinsic matrices for each frame
    :param frame_indices: Array of which frame corresponds to which 3D point
    :param camera_matrix: intrinsic camera matrix
    :return: Array of 2D projected points
    """
    # Create the pairwise extrinsic matrices based on the scalars
    new_translations = translation_scalars * frame_extrinsic_matrices[:, :, -1]
    new_extrinsics = np.hstack((frame_extrinsic_matrices[:, :, :3], new_translations))

    # Create the projection matrices
    absolute_extrinsics = np.multiply.accumulate(new_extrinsics)
    projections = np.einsum("...ij,...jk", camera_matrix, absolute_extrinsics[:, :3, :])

    # Create the array projection matrices for each point
    point_projections = projections[frame_indices]

    # Rotate points
    points_proj = np.einsum("...ij,...j", point_projections, points)

    # Normalise points
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    return points_proj


def pointAdjustmentSparsity(n_frames, n_points, frame_indices, point_indices):
    """
    Creates a sparse Jacobian for the least squares regression for points and frames

    :param n_frames: The number of frames
    :param n_points: The number of 3D points
    :param frame_indices: The frames corresponding to 2D image points
    :param point_indices: The 3D points corresponding to 2D image points
    :return: Sparse Jacobian matrix
    """
    m = frame_indices.size * 2
    n = n_frames + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(frame_indices.size)
    for s in range(1):
        A[2 * i, frame_indices + s] = 1
        A[2 * i + 1, frame_indices + s] = 1

    for s in range(3):
        A[2 * i, n_frames + point_indices * 3 + s] = 1
        A[2 * i + 1, n_frames + point_indices * 3 + s] = 1

    return A


def pointFun(parameters, frame_extrinsic_matrices, camera_matrix, n_points, frame_indices, point_indices, points_2D):
    """
    Takes a group of frame parameters and 3D points corresponding to original image 2D points and returns an array of
    the error

    :param parameters: Array of frame translation scalars followed by 3D points contiguously
    :param frame_extrinsic_matrices: Array of 4x4 extrinsic matrices
    :param camera_matrix: Camera intrinsic matrix
    :param n_points: The number of 3D points
    :param frame_indices: Array of frame indices to 2D point array
    :param point_indices: Array of 3D point indices to 2D point array
    :param points_2D: Array of corresponding 2D image points
    :return: The difference between the 2D points and projected 3D points
    """
    # Retrieve data
    n_frames = len(frame_extrinsic_matrices)
    translation_scalars = parameters[:n_frames].reshape((n_frames, 1))
    points_3D = parameters[n_frames:].reshape((n_points, 3))

    # Project points
    points_proj = project(points_3D[point_indices],
                          translation_scalars,
                          frame_extrinsic_matrices,
                          frame_indices,
                          camera_matrix)

    return (points_proj - points_2D).ravel()


def reformatPointResult(result, frame_extrinsic_matrices, n_points):
    """
    Converts the new calculated points, camera rotations and translations into usable arrays

    :param result: Least squares regression result
    :param frame_extrinsic_matrices: Array of 4x4 extrinsic matrices for each frame
    :param n_points: The number of points
    :return: a 3D cartesian point array,
            a 3D cartesian frame position array
    """
    n_frames = len(frame_extrinsic_matrices)

    points = result.x[n_frames:].reshape((n_points, 3))
    translation_scalars = result.x[:n_frames].reshape((n_frames, 1))

    # Create the extrinsic matrices based on the scalars
    new_translations = translation_scalars * frame_extrinsic_matrices[:, :, -1]
    new_extrinsics = np.hstack((frame_extrinsic_matrices[:, :, :3], new_translations))
    absolute_extrinsics = np.multiply.accumulate(new_extrinsics)

    # Calculate frame positions
    rotations = np.linalg.inv(absolute_extrinsics[:, :3, :3])
    translations = -absolute_extrinsics[:, :3, -1] * translation_scalars
    positions = np.einsum("...ij,...j", rotations, translations)

    return points, positions


def adjustPoints(frame_extrinsic_matrices, camera_intrinsic_matrix, points_3D, points_2D, frame_indices, point_indices):
    """
    Takes all the projections for the found 3D points and improves the projections

    :param frame_extrinsic_matrices: The 4x4 extrinsic matrices for each frame
    :param camera_intrinsic_matrix: The intrinsic camera matrix
    :param points_3D: The triangulated 3D points
    :param points_2D: The corresponding 2D image coordinates
    :param frame_indices: The frame corresponding to each 2D point
    :param point_indices: The 3D point corresponding to each 2D point
    :return: New 3D points from improved projections
    """
    frame_parameters = np.repeat([1], len(frame_extrinsic_matrices))

    # Concatenating frame parameters and 3D points
    parameters = np.concatenate((frame_parameters,
                                 points_3D.reshape((len(points_3D) * 3,))))

    # Applying least squares to find the optimal projections and hence 3D points
    A = pointAdjustmentSparsity(len(frame_parameters), len(points_3D), frame_indices, point_indices)
    res = least_squares(pointFun,
                        parameters,
                        jac_sparsity=A,
                        verbose=2,
                        x_scale='jac',
                        ftol=1e-4,
                        method='trf',
                        args=(frame_extrinsic_matrices,
                              camera_intrinsic_matrix,
                              len(points_3D),
                              frame_indices,
                              point_indices,
                              points_2D))

    return reformatPointResult(res, frame_extrinsic_matrices, len(points_3D))
