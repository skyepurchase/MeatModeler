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


def project(points, frame_params):
    """
    Takes an array of 3D points and corresponding camera parameters and returns the re-projected 2D points

    :param points: Array of 3D points
    :param frame_params: Array of frame parameters (rotation vectors, translation vectors, intrinsic properties)
    :return: Array of 2D projected points
    """
    # Project and normalise points
    points_proj = rotate(points, frame_params[:, :3])
    points_proj += frame_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    # Remove artifacts from the camera
    f = frame_params[:, 6]
    k1 = frame_params[:, 7]
    k2 = frame_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]

    return points_proj


def fun(parameters, n_frames, n_points, frame_indices, point_indices, points_2D):
    """
    Takes a group of frame parameters and 3D points corresponding to original image 2D points and returns an array of
    the error

    :param parameters: Array of frame parameters followed by 3D points contiguously
    :param n_frames: The number of frames
    :param n_points: The number of 3D points
    :param frame_indices: Array of frame indices to 2D point array
    :param point_indices: Array of 3D point indices to 2D point array
    :param points_2D: Array of corresponding 2D image points
    :return: The difference between the 2D points and projected 3D points
    """
    # Retrieve data
    frame_params = parameters[:n_frames * 9].reshape((n_frames, 9))
    points_3D = parameters[n_frames * 9:].reshape((n_points, 3))

    # Project points
    points_proj = project(points_3D[point_indices], frame_params[frame_indices])

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
    n = n_frames * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(frame_indices.size)
    for s in range(9):
        A[2 * i, frame_indices * 9 + s] = 1
        A[2 * i + 1, frame_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_frames * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_frames * 9 + point_indices * 3 + s] = 1

    return A


def bundleAdjustment(frame_projections, camera_matrix, points_3D, points_2D, frame_indices, point_indices):
    """
    Takes all the projections for the found 3D points and improves the projections

    :param frame_projections: The 4x3 frame projection matrices
    :param camera_matrix: The intrinsic camera matrix
    :param points_3D: The triangulated 3D points
    :param points_2D: The corresponding 2D image coordinates
    :param frame_indices: The frame corresponding to each 2D point
    :param point_indices: The 3D point corresponding to each 2D point
    :return: New 3D points from improved projections
    """
    # Converting array of projection matrices into an array rotation vectors and translation vectors
    # Creating the transposed translation vector array
    translation_vectors = np.hstack((np.expand_dims(frame_projections[:, 0, 3], axis=1),
                                     np.expand_dims(frame_projections[:, 1, 3], axis=1),
                                     np.expand_dims(frame_projections[:, 2, 3], axis=1)))

    # Finding the matrix of rotation Euler angles
    theta = np.arccos((frame_projections[:, 0, 0] + frame_projections[:, 1, 1] + frame_projections[:, 2, 2] - 1) / 2)
    sin_theta = np.sin(theta)

    # Finding the matrix of transposed unit vectors of rotation
    with np.errstate(invalid='ignore'):
        rotation_vectors_x = (frame_projections[:, 2, 1] - frame_projections[:, 1, 2]) / (2 * sin_theta)
        rotation_vectors_y = (frame_projections[:, 0, 2] - frame_projections[:, 2, 0]) / (2 * sin_theta)
        rotation_vectors_z = (frame_projections[:, 1, 0] - frame_projections[:, 0, 1]) / (2 * sin_theta)

        rotation_vectors = np.hstack((np.expand_dims(rotation_vectors_x, axis=1),
                                      np.expand_dims(rotation_vectors_y, axis=1),
                                      np.expand_dims(rotation_vectors_z, axis=1)))

        # Scaling the unit vectors by the size of the angle
        rotation_vectors = np.nan_to_num(rotation_vectors) * np.expand_dims(theta, axis=1)

    # Matrix of the individual frame parameters
    frame_parameters = np.hstack((rotation_vectors, translation_vectors))

    # Adding camera focal length
    focal_length_vector = np.repeat((camera_matrix[0, 0] + camera_matrix[1, 1]) / 2, len(frame_parameters))
    frame_parameters = np.hstack((frame_parameters, np.expand_dims(focal_length_vector, axis=1)))

    # Adding distortion (as images are undistorted these are just 1)
    frame_parameters = np.hstack((frame_parameters, np.ones((len(frame_parameters), 2))))

    # Concatenating frame parameters and 3D points
    parameters = np.hstack((frame_parameters.reshape((len(frame_parameters)*9,)),
                            points_3D.reshape((len(points_3D)*3,))))

    # Applying least squares to find the optimal projections and hence 3D points
    A = bundleAdjustmentSparsity(len(frame_parameters), len(points_3D), frame_indices, point_indices)
    res = least_squares(fun,
                        parameters,
                        jac_sparsity=A,
                        verbose=2,
                        x_scale='jac',
                        ftol=1e-4,
                        method='trf',
                        args=(len(frame_parameters), len(points_3D), frame_indices, point_indices, points_2D))

    return res.x[len(frame_parameters) * 9:].reshape((len(points_3D), 3))
