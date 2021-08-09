import time

import cv2
import itertools
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

import bundleAdjuster
from track import Track

import plotly.graph_objects as go


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
                     threshold=0.2):
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
            new_err = np.nan_to_num(err)  # If a feature wasn't trackable error is NaN
            new_err[new_err < 0] = 0  # Unknown but can result in very large negative numbers
            accumulated_error += np.average(new_err)

        # Current frame has deviated enough to be considered a key frame
        if accumulated_error > threshold * frame_grey.shape[1]:
            accumulated_error = 0

            # Recalculate points for new keyframe
            prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                        mask=None,
                                                        **feature_params)

            return True, prev_frame_grey, prev_frame_points, accumulated_error

    return False, prev_frame_grey, prev_frame_points, accumulated_error


def featureTracking(new_keyframe, prev_orb_points, prev_orb_descriptors, orb, flann_params, threshold=0.75):
    """
    Finds which features in two keyframes match

    :param new_keyframe: The keyframe to compare to the previous keyframe
    :param prev_orb_points: The previous keyframe feature points
    :param prev_orb_descriptors: The previous keyframe feature descriptors
    :param orb: An ORB feature detection object
    :param flann_params: Parameters to tune FLANN matcher
    :param threshold: Ratio threshold for FLANN matches
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
                    len(match) == 2 and match[0].distance < threshold * match[1].distance]

    left_matches = np.array([prev_orb_points[m.queryIdx].pt for m in good_matches])
    right_matches = np.array([new_points[m.trainIdx].pt for m in good_matches])

    return left_matches, right_matches, new_points, new_descriptors


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


def triangulatePoints(popped_tracks, projections, point_ID, points_2d, frame_indices, point_indices):
    """
    Generates the new 3D points from the popped_tracks and poses as well as keeping track of how the points and
    frames link together

    :param popped_tracks: The tracks that will not be updated again
    :param projections: The poses of the frames so far
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
        projection1 = projections[frame_ID1][:3, :]
        projection2 = projections[frame_ID2][:3, :]

        # Get coordinates
        coordinates = np.array(coordinates)
        left_points = coordinates[:, 0, :]
        right_points = coordinates[:, 1, :]

        # Triangulate points
        new_points = cv2.triangulatePoints(projection1, projection2, left_points.T, right_points.T).T
        new_points = new_points[:, :3] / new_points[:, -1, np.newaxis]

        # Manage bundling
        for track, point in zip(track_group, new_points):
            new_points_2d = track.get2DPoints()

            for i, point_2d in enumerate(new_points_2d):
                points_2d.append(point_2d)

                if i == len(new_points_2d) - 1:
                    # The last index must be frame_ID2 index
                    # This is a problem for comparing the last frame to the first
                    # As the first frame is 0 which cannot be reached with addition
                    frame_indices.append(frame_ID2)
                else:
                    frame_indices.append(frame_ID1 + i)

                point_indices.append(point_ID)

            point_ID += 1
            count += 1

        if points is None:
            points = new_points
        else:
            points = np.concatenate((points, new_points))

    return points, point_ID, points_2d, frame_indices, point_indices


def process(video, lk_params, feature_params, flann_params):
    """
    Takes a video of a food item and returns the 3D mesh of the food item

    :param video: The video to be converted to a 3D mesh
    :param path: The path to save images to
    :param intrinsic_matrix: The intrinsic matrix for the video camera used
    :param lk_params: Lucas-Kanade feature tracking parameters
    :param feature_params: OpenCV GoodFeaturesToTrack parameters
    :param flann_params: FLANN feature matching parameters
    :return: A 3D point cloud
    """
    cap = cv2.VideoCapture(video)
    count = 0

    # Retrieve first frame
    success, frame = cap.read()
    frame_size = frame.shape[::-1][1:]
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialise keyframe tracking
    prev_frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Initialise feature tracking
    orb = cv2.ORB_create(nfeatures=20000)
    prev_orb_points, prev_orb_descriptors = orb.detectAndCompute(prev_frame_grey, None)

    # # Initialise point tracking
    # tracks = []
    # popped_tracks = []
    # prev_keyframe_ID = 0
    # keyframe_ID = 1
    #
    # # Initialise bundling
    # points_2d = []
    # frame_indices = []
    # point_indices = []
    # point_ID = 0
    #
    # toc = time.time()
    #
    # print("Initialisation complete.")
    # print(toc - tic, "seconds.\n")
    #
    # print("Finding points...")
    #
    # tic = time.time()

    # Processing loop
    # Initialise calibration
    usable_frames = []
    img_points = []
    calibration_obj_points = []

    calibration_objp = np.zeros((12, 3), np.float32)
    stereo_objp = np.zeros((12, 3), np.float32)

    # Let z=0 for all chessboard corners
    calibration_objp[:, :2] = np.mgrid[0:4, 0:3].T.reshape(-1, 2) * 2

    # Let y = 0 for all chessboard corners
    # Depth is z and height is y. The chessboard is assumed to be at constant height not depth
    stereo_objp[:, 0] = calibration_objp[:, 0]
    stereo_objp[:, 1] = calibration_objp[:, 2]
    stereo_objp[:, 2] = calibration_objp[:, 1]

    print("Finding usable keyframes", end="...")
    while success:
        frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)
        count += 1

        is_keyframe, prev_frame_grey, prev_frame_points, accumulative_error = keyframeTracking(frame_grey,
                                                                                               prev_frame_grey,
                                                                                               prev_frame_points,
                                                                                               accumulative_error,
                                                                                               lk_params,
                                                                                               feature_params,
                                                                                               threshold=0.1)

        hasChessboard, corners = cv2.findChessboardCorners(frame_grey, (4, 3))

        if is_keyframe and hasChessboard:
            usable_frames.append(frame_grey)
            img_points.append(corners)
            calibration_obj_points.append(calibration_objp)

            L_matches, R_matches, prev_orb_points, prev_orb_descriptors = featureTracking(frame_grey,
                                                                                          prev_orb_points,
                                                                                          prev_orb_descriptors,
                                                                                          orb,
                                                                                          flann_params)



            # # Calculate matches
            # print("\nFinding matches", end="...")
            # L_matches, R_matches, prev_orb_points, prev_orb_descriptors = featureTracking(frame_grey,
            #                                                                               prev_orb_points,
            #                                                                               prev_orb_descriptors,
            #                                                                               orb,
            #                                                                               flann_params)
            # print("found", len(L_matches))
            #
            # # Pose estimation
            # print("Finding inliers", end="...")
            # L_points, R_points, right_extrinsic, pairwise_extrinsic, projection = poseEstimation(L_matches,
            #                                                                                      R_matches,
            #                                                                                      left_extrinsic,
            #                                                                                      intrinsic_matrix)
            # print("found", len(L_points))
            #
            # projections.append(projection)
            # extrinsic_matrices.append(pairwise_extrinsic)
            #
            # # Manage tracks
            # print("Grouping points", end="...")
            # new_popped_tracks, tracks = pointTracking(tracks,
            #                                           prev_keyframe_ID,
            #                                           L_points,
            #                                           keyframe_ID,
            #                                           R_points)
            # popped_tracks += new_popped_tracks
            # print(len(popped_tracks), "potential points")
            #
            # # Update variables
            # left_extrinsic = right_extrinsic  # Right keyframe now becomes the left keyframe
            # prev_keyframe_ID = keyframe_ID
            # keyframe_ID += 1

        success, frame = cap.read()

    print("found", len(img_points), end="\n\n")

    print("Calibrating camera", end="...")

    success, intrinsic_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(calibration_obj_points,
                                                                              img_points,
                                                                              frame_size,
                                                                              None,
                                                                              None)

    print("calibrated\n\n")

    print("Calculating projections", end="...")
    extrinsics = np.array([np.hstack((cv2.Rodrigues(rvec)[0], tvec)) for rvec, tvec in zip(rvecs, tvecs)])
    projections = np.einsum("ij,...jk", intrinsic_matrix, extrinsics)
    print("done\n\n")

    positions = bundleAdjuster.rotate(-np.array(tvecs).reshape((len(tvecs), 1, 3)),
                                      -np.array(rvecs).reshape((len(rvecs), 1, 3))).reshape((len(rvecs), 3))
    looking_at = bundleAdjuster.rotate(np.repeat([[0, 0, 1]], len(rvecs)).reshape(len(rvecs), 1, 3),
                                       -np.array(rvecs).reshape((len(rvecs), 1, 3))).reshape((len(rvecs)), 3)

    fig = go.Figure()
    positions = np.array(positions)

    fig.add_trace(go.Scatter3d(x=positions[:, 0],
                               y=positions[:, 2],
                               z=positions[:, 1],
                               mode="text+lines+markers",
                               name="Camera positions",
                               textposition="top center",
                               text=[str(i) for i in range(len(positions))]))

    fig.add_trace(go.Scatter3d(x=calibration_objp[:, 0],
                               y=calibration_objp[:, 2],
                               z=calibration_objp[:, 1],
                               mode="text+lines+markers",
                               name="Chessboard positions",
                               textposition="top center",
                               text=[str(i) for i in range(len(calibration_objp))]))

    fig.add_trace(go.Scatter3d(x=looking_at[:, 0] + positions[:, 0],
                               y=looking_at[:, 2] + positions[:, 2],
                               z=looking_at[:, 1] + positions[:, 1],
                               mode="text+markers",
                               name="Looking_at",
                               textposition="top center",
                               text=[str(i) for i in range(len(positions))]))

    fig.show()

    exit()

    # # Add the remaining tracks which are implicitly popped
    # popped_tracks += tracks
    #
    # # Include the points in the tracks not popped at the end
    # print("Triangulating points", end="...")
    # points, point_ID, points_2d, frame_indices, point_indices = triangulatePoints(popped_tracks,
    #                                                                               projections,
    #                                                                               point_ID,
    #                                                                               points_2d,
    #                                                                               frame_indices,
    #                                                                               point_indices)
    # print("done")
    #
    # toc = time.time()
    #
    # print(len(tracks), "points found.")
    # print(toc - tic, "seconds.\n")
    #
    # print("adjusting points...")
    # tic = time.time()
    #
    # extrinsics = np.array(extrinsic_matrices)
    # extrinsics = np.array(list(itertools.accumulate(extrinsics, lambda n, m: np.dot(n, m))))
    # adjusted_points, adjusted_positions = bundleAdjuster.adjustPoints(extrinsics,
    #                                                                   intrinsic_matrix,
    #                                                                   points,
    #                                                                   np.array(points_2d),
    #                                                                   np.array(frame_indices),
    #                                                                   np.array(point_indices))
    #
    # toc = time.time()
    # print("adjustment complete.")
    # print(toc - tic, "seconds.\n")
    #
    # print("Saving point cloud...")
    # tic = time.time()
    #
    # filename = path + "Cloud.ply"
    # cloud = PyntCloud(pd.DataFrame(
    #     data=adjusted_points,
    #     columns=['x', 'y', 'z']
    # ))
    # cloud.to_file(filename)
    #
    # toc = time.time()
    # print("Point cloud saved.")
    # print(tic - toc)
