import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


def rotate(points, rot_vecs):
    """
    Takes an array of 3D points and an array of camera rotation vectors and returns the rotated points

    :param points: Array of 3D points
    :param rot_vecs: Array of Euler-Rodrigues rotation vectors
    :return: Array of 3D points
    """
    # The angle of rotation is the magnitude of the rotation vectors
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]

    # Normalising vectors
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)

    # Rodrigues method for rotation
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, frame_params, camera_matrix):
    """
    Takes an array of 3D points and corresponding camera parameters and returns the re-projected 2D points

    :param points: Array of 3D points
    :param frame_params: Array of frame parameters (rotation vectors, translation vectors, intrinsic properties)
    :param camera_matrix: intrinsic camera matrix
    :return: Array of 2D projected points
    """
    # Rotate points
    points_proj = rotate(points, frame_params[:, :3])

    # Translate points
    points_proj += frame_params[:, 3:6]

    # Convert to world coordinates
    points_proj = np.einsum("ij,...j", camera_matrix, points_proj)

    # Normalise points
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    return points_proj


def pointFun(parameters, camera_matrix, n_frames, n_points, frame_indices, point_indices, points_2D):
    """
    Takes a group of frame parameters and 3D points corresponding to original image 2D points and returns an array of
    the error

    :param parameters: Array of frame parameters followed by 3D points contiguously
    :param camera_matrix: Camera intrinsic matrix
    :param n_frames: The number of frames
    :param n_points: The number of 3D points
    :param frame_indices: Array of frame indices to 2D point array
    :param point_indices: Array of 3D point indices to 2D point array
    :param points_2D: Array of corresponding 2D image points
    :return: The difference between the 2D points and projected 3D points
    """
    # Retrieve data
    frame_params = parameters[:n_frames * 6].reshape((n_frames, 6))
    points_3D = parameters[n_frames * 6:].reshape((n_points, 3))

    # Project points
    points_proj = project(points_3D[point_indices], frame_params[frame_indices], camera_matrix)

    return (points_proj - points_2D).ravel()


def bundleAdjustmentSparsity(n_frames, n_points, frame_indices, point_indices):
    """
    Creates a sparse Jacobian for the least squares regression

    :param n_frames: The number of frames
    :param n_points: The number of 3D points
    :param frame_indices: The frames corresponding to 2D image points
    :param point_indices: The 3D points corresponding to 2D image points
    :return: Sparse Jacobian matrix
    """
    m = frame_indices.size * 2
    n = n_frames * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(frame_indices.size)
    for s in range(6):
        A[2 * i, frame_indices * 6 + s] = 1
        A[2 * i + 1, frame_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_frames * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_frames * 6 + point_indices * 3 + s] = 1

    return A


def findPositions(parameters, n_frames):
    """
    Calculates the positions of the cameras based on their extrinsic matrices

    :param parameters: Contiguous array of frame parameters (n_frames * 6,)
    :param n_frames: The number of frames present in the parameters
    :return: An array of 3D cartesian positions (n_frames, 3)
    """
    transformations = parameters.reshape((n_frames, 6))
    inv_rotations = -transformations[:, :3]
    translations = -transformations[:, 3:]
    positions = rotate(translations, inv_rotations)
    return positions


def reformatResult(result, n_frames, n_points):
    """
    Converts the new calculated points, camera rotations and translations into usable arrays

    :param result: Least squares regression result
    :param n_frames: The number of frames
    :param n_points: The number of points
    :return: a 3D cartesian point array,
            a 3D cartesian frame position array
    """
    points = result.x[n_frames * 6:].reshape((n_points, 3))

    # Pose matrix transforms world coordinates to camera coordinates
    positions = findPositions(result.x[:n_frames * 6], n_frames)

    return points, positions


def frameParameters(frame_projections):
    """
    Converts 4x4 frame projection matrices into 2 contiguous row vectors using Euler-Rodrigues rotation vectors

    :param frame_projections: An array of 4x4 projection matrices
    :return: An array of rotation and translation vectors
    """
    # Converting array of projection matrices into an array rotation vectors and translation vectors
    # Creating the transposed translation vector array
    translation_vectors = frame_projections[:, :3, 3]

    # Finding the matrix of rotation Euler angles
    theta = np.arccos((frame_projections[:, 0, 0] + frame_projections[:, 1, 1] + frame_projections[:, 2, 2] - 1) / 2)
    sin_theta = np.sin(theta)

    # Finding the matrix of transposed unit vectors of rotation
    with np.errstate(invalid='ignore'):
        rotation_vectors_x = (frame_projections[:, 2, 1] - frame_projections[:, 1, 2]) / (2 * sin_theta)
        rotation_vectors_y = (frame_projections[:, 0, 2] - frame_projections[:, 2, 0]) / (2 * sin_theta)
        rotation_vectors_z = (frame_projections[:, 1, 0] - frame_projections[:, 0, 1]) / (2 * sin_theta)

        rotation_vectors = np.vstack((rotation_vectors_x, rotation_vectors_y, rotation_vectors_z)).T

        # Scaling the unit vectors by the size of the angle
        rotation_vectors = np.nan_to_num(rotation_vectors) * np.expand_dims(theta, axis=1)

    # Matrix of the individual frame parameters
    return np.hstack((rotation_vectors, translation_vectors))


def adjustPoints(frame_projections, camera_matrix, points_3D, points_2D, frame_indices, point_indices):
    """
    Takes all the projections for the found 3D points and improves the projections

    :param frame_projections: The 4x3 frame projection matrices (not modified by the intrinsic matrix)
    :param camera_matrix: The intrinsic camera matrix
    :param points_3D: The triangulated 3D points
    :param points_2D: The corresponding 2D image coordinates
    :param frame_indices: The frame corresponding to each 2D point
    :param point_indices: The 3D point corresponding to each 2D point
    :return: New 3D points from improved projections
    """
    frame_parameters = frameParameters(frame_projections)

    # Concatenating frame parameters and 3D points
    parameters = np.hstack((frame_parameters.reshape((len(frame_parameters)*6,)),
                            points_3D.reshape((len(points_3D)*3,))))

    # Applying least squares to find the optimal projections and hence 3D points
    A = bundleAdjustmentSparsity(len(frame_parameters), len(points_3D), frame_indices, point_indices)
    res = least_squares(pointFun,
                        parameters,
                        jac_sparsity=A,
                        verbose=2,
                        x_scale='jac',
                        ftol=1e-4,
                        method='trf',
                        args=(camera_matrix,
                              len(frame_parameters),
                              len(points_3D),
                              frame_indices,
                              point_indices,
                              points_2D))

    return reformatResult(res, len(frame_parameters), len(points_3D))
