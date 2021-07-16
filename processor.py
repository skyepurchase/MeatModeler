import time

import cv2
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

import bundleAdjuster
from track import Track


def increaseContrast(frame):
    """
    Increases the contrast of the grey scale images by applying CLAHE to the luminance

    :param frame: The frame to be editted
    :return: The increased contrast image
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_out = clahe.apply(l)
    lab_out = cv2.merge((l_out, a, b))

    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


def calibrate(images, corner_dims=(7, 7)):
    """
    Takes specific chess board images and calibrates the camera appropriately

    :param images: A list of different openCV image objects of a known chessboard
    :param corner_dims: A tuple the dimensions of the chessboard corners (standard board is (7, 7) and default input)
    :return: The intrinsic property matrix,
            The distortion coefficients
    """
    # Prepare chessboard 3D points
    x, y = corner_dims
    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object and image points from all images
    obj_points = []
    img_points = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        success, corners = cv2.findChessboardCorners(gray, corner_dims, None)

        # If found, add object points, image points
        if success:
            obj_points.append(objp)
            img_points.append(corners)

    img = images[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    success, matrix, distortion, _, _ = cv2.calibrateCamera(obj_points,
                                                            img_points,
                                                            gray.shape[::-1],
                                                            None,
                                                            None)

    if success:
        return matrix, distortion

    return None


def undistortFrame(frame, camera_matrix, distortion_coefficients):
    """
    Takes a frame and removes the distortion cause by the camera

    :param frame: Frame with distortion
    :param camera_matrix: The intrinsic matrix of the camera
    :param distortion_coefficients: The distortion coefficients of the camera
    :return: Frame without distortion, cropped to ROI
    """
    height, width = frame.shape[:2]

    # Find the new camera matrix for the specific frame
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                               distortion_coefficients,
                                                               (width, height),
                                                               1,
                                                               (width, height))

    # Utilise this to get an undistorted frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, optimal_camera_matrix)

    # Crop the edges
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y + h, x:x + w]

    return undistorted_frame


def keyframeTracking(frame_grey, prev_frame_grey, prev_frame_points, accumulated_error, lk_params, feature_params,
                     threshold=0.1):
    """
    Determines whether a given frame is a keyframe for further analysis

    :param frame_grey: The frame to be analysed in greyscale
    :param prev_frame_grey: The previous frame in greyscale
    :param prev_frame_points: The previous frame features to track
    :param accumulated_error: The current error distance from keyframe points
    :param lk_params: Lucas-Kanade parameters
    :param feature_params: GoodFeaturesToTrack parameters
    :param threshold: Proportion of the frame width considered significant
    :return: A boolean value on whether the frame was a keyframe,
            The new previous grey frame,
            The new previous frame points,
            The new accumulated error
    """
    # Compare the last key frame to current key frame
    p, st, err = cv2.calcOpticalFlowPyrLK(prev_frame_grey,
                                          frame_grey,
                                          prev_frame_points,
                                          None,
                                          **lk_params)

    # Keep only matching points
    if p is not None:
        good_new = p[st == 1]

        # Update previous data
        prev_frame_grey = frame_grey
        prev_frame_points = good_new.reshape(-1, 1, 2)

        # If possible increase the accumulative error between frames
        if err is not None:
            accumulated_error += np.average(err)

        # Current frame has deviated enough to be considered a key frame
        if accumulated_error > threshold * frame_grey.shape[1]:
            accumulated_error = 0

            # Recalculate points for new keyframe
            prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                        mask=None,
                                                        **feature_params)

            return True, prev_frame_grey, prev_frame_points, accumulated_error

    return False, prev_frame_grey, prev_frame_points, accumulated_error


def featureTracking(new_keyframe, prev_orb_points, prev_orb_descriptors, orb, flann_params):
    """
    Finds which features in two keyframes match

    :param new_keyframe: The keyframe to compare to the previous keyframe
    :param prev_orb_points: The previous keyframe feature points
    :param prev_orb_descriptors: The previous keyframe feature descriptors
    :param orb: An ORB feature detection object
    :param flann_params: Parameters to tune FLANN matcher
    :return: List of left frame matched Keypoints,
            List of right frame matched Keypoints,
            The new previous keyframe feature points,
            The new previous keyframe feature descriptors
    """
    # Get new points and descriptors
    new_points, new_descriptors = orb.detectAndCompute(new_keyframe, None)

    # FLANN based approach to find matches
    flann = cv2.FlannBasedMatcher(flann_params, {})
    matches = flann.knnMatch(prev_orb_descriptors, new_descriptors, k=2)

    # Find which points can be considered new
    good_matches = [match[0] for match in matches if
                    len(match) == 2 and match[0].distance < 0.8 * match[1].distance]

    left_matches = np.array([prev_orb_points[m.queryIdx].pt for m in good_matches])
    right_matches = np.array([new_points[m.trainIdx].pt for m in good_matches])

    return left_matches, right_matches, new_points, new_descriptors


def poseEstimation(left_frame_points, right_frame_points, origin_to_left, camera_matrix):
    """
    Takes the matches between two frames and the transformation between origin and left frame coordinates and finds
    the transformation between origin and right frame coordinates

    :param left_frame_points: Undistorted matched points from the left frame
    :param right_frame_points: Undistorted matched points from the right frame
    :param origin_to_left: 4x4 matrix converting origin coordinates to left frame coordinates
    :param camera_matrix: The intrinsic matrix of the camera
    :return: The used left points,
            The used right points,
            The corresponding 3D points,
            The new previous pose matrix
    """
    # Find essential matrix and inliers
    essential_left_to_right, mask_E = cv2.findEssentialMat(right_frame_points,
                                                           left_frame_points,
                                                           camera_matrix)

    # Use the essential matrix and inliers to find the pose and new inliers
    _, R_left_to_right, t_left_to_right, mask_RP = cv2.recoverPose(essential_left_to_right,
                                                                   right_frame_points,
                                                                   left_frame_points,  # Potentially swap
                                                                   camera_matrix,
                                                                   mask=mask_E)

    # Create the 4x3 pose matrix from rotation and translation
    transform_left_to_right = np.hstack([R_left_to_right, t_left_to_right])

    # Convert to homogeneous 4x4 transformation matrix
    transform_left_to_right = np.vstack((transform_left_to_right, np.array([0, 0, 0, 1])))

    # Take world coordinates to left frame then to right frame
    transform_origin_to_right = np.matmul(transform_left_to_right, origin_to_left)

    # The real world pose
    pose = np.dot(camera_matrix, transform_origin_to_right[:3])

    # Usable points
    usable_left_points = left_frame_points[mask_RP[:, 0] == 1]
    usable_right_points = right_frame_points[mask_RP[:, 0] == 1]

    return usable_left_points, usable_right_points, transform_origin_to_right, pose


def pointTracking(tracks, prev_keyframe_ID, feature_points, keyframe_ID, correspondents):
    """
    Checks through the current tracks and updates them based on the provided matches

    :param tracks: Current tracks
    :param prev_keyframe_ID: The identity number of the previous keyframe
    :param feature_points: The feature point matches from the previous keyframe
    :param keyframe_ID: The identity number of the current keyframe
    :param correspondents: The corresponding feature match
    :return: The tracks to be processed,
            Continuing tracks
    """

    new_tracks = []
    updated_tracks = []
    popped_tracks = []

    # For each match check if this feature already exists
    for feature_point, correspondent in zip(feature_points, correspondents):
        # Convert to tuples
        feature_point = (feature_point[0], feature_point[1])
        correspondent = (correspondent[0], correspondent[1])

        is_new_track = True

        for track in tracks:
            # If the current point matches the track's last point then they reference the same feature
            prior_point = track.getLastPoint()

            # So update the track
            if feature_point == prior_point:
                track.update(keyframe_ID, correspondent)
                is_new_track = False
                break

        # Feature was not found elsewhere
        if is_new_track:
            new_track = Track(prev_keyframe_ID,
                              feature_point,
                              keyframe_ID,
                              correspondent)
            new_tracks.append(new_track)

    for track in tracks:
        if track.wasUpdated():
            track.reset()
            updated_tracks.append(track)
        else:
            popped_tracks.append(track)

    # Add new tracks
    updated_tracks += new_tracks

    return popped_tracks, updated_tracks


def triangulation(first_pose, last_pose, left_points, right_points, tolerance=3.e-5):
    """"""
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    # Create array of triangulated points
    x = np.empty((4, len(left_points)))
    x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
    x_status = np.empty(len(left_points), dtype=int)

    # Initialize C matrices
    C1 = np.array(-np.eye(2, 3))
    C2 = np.array(-np.eye(2, 3))

    for xi in range(len(left_points)):
        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = left_points[xi, :]
        C2[:, 2] = right_points[xi, :]

        # Build A matrix
        A[0:2, :] = C1.dot(first_pose[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C2.dot(last_pose[0:3, 0:3])  # C2 * R2

        # Build b vector
        b[0:2, :] = C1.dot(first_pose[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C2.dot(last_pose[0:3, 3:4])  # C2 * t2
        b *= -1

        # Init depths
        d1 = d2 = 1.

        for i in range(10):  # Hartley suggests 10 iterations at most
            # Solve for x vector
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d1_new = first_pose[2, :].dot(x[:, xi])
            d2_new = last_pose[2, :].dot(x[:, xi])

            # Set status
            x_status[xi] = (i < 10 and  # points should have converged by now
                            (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras

            if d1_new <= 0:
                x_status[xi] -= 1

            if d2_new <= 0:
                x_status[xi] -= 2

            if abs(d1_new - d1) <= tolerance and \
                    abs(d2_new - d2) <= tolerance:
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d1_new
            A[2:4, :] *= 1 / d2_new
            b[0:2, :] *= 1 / d1_new
            b[2:4, :] *= 1 / d2_new

            # Update depths
            d1 = d1_new
            d2 = d2_new

    # Return only the points in front of both cameras
    x = x[:4, :].T
    x = x[:, :3] / x[:, -1][:, None]
    x = x[np.expand_dims(x_status, axis=1)[:, 0] == 1]

    return x, (np.expand_dims(x_status, axis=1)[:, 0] == 1)


def managePoints(popped_tracks, poses, point_ID, points_2d, frame_indices, point_indices):
    """
    Generates the new 3D points from the popped_tracks and poses as well as keeping track of how the points and
    frames link together

    :param popped_tracks: The tracks that will not be updated again
    :param poses: The poses of the frames so far
    :param point_ID: The current 3D point identification number
    :param points_2d: The 2D image points analysed so far
    :param frame_indices: The index of the frame relating to each 2D point
    :param point_indices: The index of the 3D point relating to each 2D point
    :return: new 3D points,
            new 3D point identification number,
            new 2D point array
            new frame index array
            new 3D point index array
    """
    # Join together all the points and tracks for pairs of frames
    frame_pairs = {}
    for track in popped_tracks:
        frame_ID1, frame_ID2, left, right = track.getTriangulationData()
        pair = [left, right]
        identifier = str(frame_ID1) + "-" + str(frame_ID2)

        if identifier in frame_pairs:
            track_group, coordinates = frame_pairs.get(identifier)
            track_group.append(track)
            coordinates.append(pair)
            frame_pairs[identifier] = (track_group, coordinates)
        else:
            track_group = [track]
            coordinates = [pair]
            frame_pairs[identifier] = (track_group, coordinates)

    points = None
    count = 0

    # Triangulation
    for identifier, (track_group, coordinates) in frame_pairs.items():
        frames = identifier.split("-")
        frame_ID1 = int(frames[0])
        frame_ID2 = int(frames[1])

        # Get poses
        pose1 = poses[frame_ID1][:3, :]
        pose2 = poses[frame_ID2][:3, :]

        # Get coordinates
        coordinates = np.array(coordinates)
        left_points = coordinates[:, 0, :]
        right_points = coordinates[:, 1, :]

        # Triangulate points
        new_points, used = triangulation(pose1, pose2, left_points, right_points)

        # Manage bundling
        track_group = np.array(track_group)
        used_tracks = track_group[used]
        for track, point in zip(used_tracks, new_points):
            new_points_2d = track.get2DPoints()

            for i, point_2d in enumerate(new_points_2d):
                points_2d.append(point_2d)
                frame_indices.append(frame_ID1 + i)
                point_indices.append(point_ID)

            point_ID += 1
            count += 1

        if points is None:
            points = new_points
        else:
            points = np.concatenate((points, new_points))

    return points, point_ID, points_2d, frame_indices, point_indices


def process(video, path, intrinsic_matrix, distortion_coefficients, lk_params, feature_params, flann_params):
    """
    Takes a video of a food item and returns the 3D mesh of the food item

    :param video: The video to be converted to a 3D mesh
    :param path: The path to save images to
    :param intrinsic_matrix: The intrinsic matrix for the video camera used
    :param distortion_coefficients: The disrotion coefficients for the video camera used
    :param lk_params: Lucas-Kanade feature tracking parameters
    :param feature_params: OpenCV GoodFeaturesToTrack parameters
    :param flann_params: FLANN feature matching parameters
    :return: A 3D point cloud
    """
    print("Initialising...")
    tic = time.time()
    orb = cv2.ORB_create(nfeatures=20000)

    cap = cv2.VideoCapture(video)

    # Retrieve first frame
    _, start_frame = cap.read()
    start_frame = undistortFrame(start_frame, intrinsic_matrix, distortion_coefficients)

    # Initialise keyframe tracking
    prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Initialise feature tracking
    prev_orb_points, prev_orb_descriptors = orb.detectAndCompute(prev_frame_grey, None)

    # Initialise pose estimation
    origin_to_left = np.eye(4, 4)  # The first keyframe is at origin and left of next frame
    original_pose = np.dot(intrinsic_matrix, origin_to_left[:3])  # But needs to be placed into world coordinates
    poses = [original_pose]  # The first keyframe is added
    transforms = [origin_to_left]

    # Initialise point tracking
    tracks = []
    prev_keyframe_ID = 0
    keyframe_ID = 1

    # Initialise triangulation
    points = None

    # Initialise bundling
    points_2d = []
    frame_indices = []
    point_indices = []
    point_ID = 0
    toc = time.time()
    print("Initialisation complete.")
    print(toc - tic, "seconds.\n")

    print("Finding points...")
    tic = time.time()
    # Processing loop
    success, frame = cap.read()

    while success:
        frame = undistortFrame(frame, intrinsic_matrix, distortion_coefficients)
        frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)

        success, prev_frame_grey, prev_frame_points, accumulative_error = keyframeTracking(frame_grey,
                                                                                           prev_frame_grey,
                                                                                           prev_frame_points,
                                                                                           accumulative_error,
                                                                                           lk_params,
                                                                                           feature_params)

        if success:
            # Calculate matches
            L_matches, R_matches, prev_orb_points, prev_orb_descriptors = featureTracking(frame_grey,
                                                                                          prev_orb_points,
                                                                                          prev_orb_descriptors,
                                                                                          orb,
                                                                                          flann_params)

            # Pose estimation
            L_points, R_points, origin_to_right, pose = poseEstimation(L_matches,
                                                                       R_matches,
                                                                       origin_to_left,
                                                                       intrinsic_matrix)

            poses.append(pose)
            transforms.append(origin_to_right)

            # Manage tracks
            popped_tracks, tracks = pointTracking(tracks,
                                                  prev_keyframe_ID,
                                                  L_points,
                                                  keyframe_ID,
                                                  R_points)

            # Manage the 3D points including triangulation and preparation for adjustment
            new_points, point_ID, points_2d, frame_indices, point_indices = managePoints(popped_tracks,
                                                                                         poses,
                                                                                         point_ID,
                                                                                         points_2d,
                                                                                         frame_indices,
                                                                                         point_indices)

            if points is None:
                points = new_points
            else:
                points = np.concatenate((points, new_points))

            # Update variables
            origin_to_left = origin_to_right  # Right keyframe now becomes the left keyframe
            prev_keyframe_ID = keyframe_ID
            keyframe_ID += 1

        success, frame = cap.read()

    # Include the points in the tracks not popped at the end
    new_points, point_ID, points_2d, frame_indices, point_indices = managePoints(tracks,
                                                                                 poses,
                                                                                 point_ID,
                                                                                 points_2d,
                                                                                 frame_indices,
                                                                                 point_indices)

    points = np.concatenate((points, new_points))

    toc = time.time()
    print("Points found.")
    print(toc - tic, "seconds.\n")

    print("adjusting points...")
    tic = time.time()
    points = bundleAdjuster.bundleAdjustment(np.array(transforms),
                                             intrinsic_matrix,
                                             points,
                                             np.array(points_2d),
                                             np.array(frame_indices),
                                             np.array(point_indices))
    toc = time.time()
    print("adjustment complete.")
    print(toc - tic, "seconds.\n")

    # Convert points to world coordinates
    points = np.einsum("ij,...j", intrinsic_matrix, points)

    filename = path + "Cloud.ply"
    cloud = PyntCloud(pd.DataFrame(
        data=points,
        columns=['x', 'y', 'z']
    ))
    cloud.to_file(filename)
