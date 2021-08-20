import cv2
import numpy as np
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
    points_proj = points_proj[:, :2] / points_proj[:, -1, np.newaxis]

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


def frameParameters(frame_extrinsic_matrices):
    """
    Converts 4x4 extrinsic matrices into 2 contiguous row vectors using Euler-Rodrigues rotation vectors

    :param frame_extrinsic_matrices: An array of 4x4 extrinsic matrices fro each frame
    :return: An array of rotation and translation vectors
    """
    # Converting array of projection matrices into an array rotation vectors and translation vectors
    # Creating the transposed translation vector array
    translation_vectors = frame_extrinsic_matrices[:, :3, 3]

    # Finding the matrix of rotation Euler angles
    theta = np.arccos((frame_extrinsic_matrices[:, 0, 0] +
                       frame_extrinsic_matrices[:, 1, 1] +
                       frame_extrinsic_matrices[:, 2, 2] - 1) / 2)
    sin_theta = np.sin(theta)

    # Finding the matrix of transposed unit vectors of rotation
    with np.errstate(invalid='ignore'):
        rotation_vectors_x = (frame_extrinsic_matrices[:, 2, 1] - frame_extrinsic_matrices[:, 1, 2]) / (2 * sin_theta)
        rotation_vectors_y = (frame_extrinsic_matrices[:, 0, 2] - frame_extrinsic_matrices[:, 2, 0]) / (2 * sin_theta)
        rotation_vectors_z = (frame_extrinsic_matrices[:, 1, 0] - frame_extrinsic_matrices[:, 0, 1]) / (2 * sin_theta)

        rotation_vectors = np.vstack((rotation_vectors_x, rotation_vectors_y, rotation_vectors_z)).T

        # Scaling the unit vectors by the size of the angle
        rotation_vectors = np.nan_to_num(rotation_vectors) * np.expand_dims(theta, axis=1)

    # Matrix of the individual frame parameters
    return np.hstack((rotation_vectors, translation_vectors)).reshape((len(frame_extrinsic_matrices) * 6,))


def reformatPointResult(result, n_frames, n_points):
    """
    Converts the new calculated points, camera rotations and translations into usable arrays

    :param result: Least squares regression result
    :param n_frames: The number of frames
    :param n_points: The number of points
    :return: a 3D cartesian point array,
            a 3D cartesian frame position array
    """
    points = result.x[n_frames * 6:].reshape((n_points, 3))
    frames = result.x[:n_frames * 6].reshape((n_frames, 6))

    rvecs = frames[:, :3]
    tvecs = frames[:, 3:]

    extrinsics = [np.vstack((np.hstack((cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1))),
                             [0, 0, 0, 1]))
                  for rvec, tvec in zip(rvecs, tvecs)]

    return points, extrinsics


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
    frame_parameters = frameParameters(frame_extrinsic_matrices)

    # Concatenating frame parameters and 3D points
    parameters = np.hstack((frame_parameters,
                            points_3D.reshape((len(points_3D) * 3,))))

    # Applying least squares to find the optimal projections and hence 3D points
    A = pointAdjustmentSparsity(len(frame_extrinsic_matrices), len(points_3D), frame_indices, point_indices)
    res = least_squares(pointFun,
                        parameters,
                        jac_sparsity=A,
                        verbose=2,
                        x_scale='jac',
                        ftol=1e-4,
                        method='trf',
                        args=(camera_intrinsic_matrix,
                              len(frame_extrinsic_matrices),
                              len(points_3D),
                              frame_indices,
                              point_indices,
                              points_2D))

    return reformatPointResult(res, len(frame_extrinsic_matrices), len(points_3D))


def reformatPoseResult(result, n_frames):
    frame_params = result.x.reshape((n_frames, 6))
    rvecs = frame_params[:, :3]
    tvecs = frame_params[:, 3:6].reshape((n_frames, 3, 1))
    extrinsic_matrices = [np.hstack((cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1))) for rvec, tvec in zip(rvecs, tvecs)]

    return extrinsic_matrices


def poseFun(parameters, camera_intrinsic_matrix, n_frames, frame_indices, point_indices, points_3D, points_2D):
    frame_params = parameters.reshape((n_frames, 6))

    points_proj = project(points_3D[point_indices], frame_params[frame_indices], camera_intrinsic_matrix)

    return (points_proj - points_2D).ravel()


def adjustPose(frame_extrinsic_matrices, camera_intrinsic_matrix, points_2D):
    # points_2D has n_frame copies of the chessboard
    n_frames = len(frame_extrinsic_matrices)
    pattern_size = int(len(points_2D) / n_frames)

    # Setup up the stationary chessboard 3D points
    points_3D = np.zeros((pattern_size, 3), np.float32)
    grid = np.mgrid[0:4, 0:3].T.reshape(-1, 2) * 2
    points_3D[:, 0] = grid[:, 0]
    points_3D[:, 2] = grid[:, 1]

    # Generating frame and point indices
    frame_indices = np.repeat(np.arange(n_frames), pattern_size)
    point_indices = np.repeat([np.arange(pattern_size)], n_frames, axis=0).reshape(pattern_size * n_frames)

    parameters = frameParameters(frame_extrinsic_matrices)

    # A = pointAdjustmentSparsity(n_frames, pattern_size, frame_indices, point_indices)
    res = least_squares(poseFun,
                        parameters,
                        verbose=2,
                        ftol=1e-4,
                        args=(camera_intrinsic_matrix,
                              n_frames,
                              frame_indices,
                              point_indices,
                              points_3D,
                              points_2D))

    return reformatPoseResult(res, n_frames)
