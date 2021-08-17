import time

import cv2
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud

import bundleAdjuster
from track import Track


def increaseContrast(frame):
    """
    Increases the contrast of the frames by applying CLAHE to the luminance

    :param frame: The frame to be editted
    :return: The increased contrast image
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_out = clahe.apply(l)
    lab_out = cv2.merge((l_out, a, b))

    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


def calibrate(img_points, size, pattern_shape):
    """
    Takes specific chess board images and calibrates the camera appropriately

    :param img_points: A list of the corners for each frame
    :param size: The frame size
    :param pattern_shape: A tuple the dimensions of the chessboard corners (standard board is (7, 7) and default input)
    :return: The intrinsic property matrix,
            The distortion coefficients
    """
    # Prepare chessboard 3D points
    x, y = pattern_shape
    objp = np.zeros((x * y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    obj_points = []

    for _ in img_points:
        obj_points.append(objp)

    success, matrix, distortion, _, _ = cv2.calibrateCamera(obj_points,
                                                            img_points,
                                                            size,
                                                            None,
                                                            None)

    if success:
        return matrix, distortion

    return None


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

    prev_matches = np.array([prev_orb_points[m.queryIdx].pt for m in good_matches])
    curr_matches = np.array([new_points[m.trainIdx].pt for m in good_matches])

    return prev_matches, curr_matches, new_points, new_descriptors


def poseEstimation(corners, image, pattern_shape, side_length, camera_intrinsic_matrix, distortion_coefficients):
    """
    Finds the frame pose from the coordinates of the identified chess pattern, calculating the extrinsic and
    projection matrices

    :param corners: The identified chess pattern coordinates from OpenCV findChessboardCorners
    :param image: The greyscale image corresponding to the identified corners
    :param pattern_shape: The shape as a tuple (width, height) of the chess pattern
    :param side_length: The real world side length of each square
    :param camera_intrinsic_matrix: The calculated intrinsic camera matrix
    :param distortion_coefficients: The calculated distortion coefficients
    :return: The rotation vector
            The translation vector
            The extrinsic matrix,
            The projection matrix
    """
    # Generate object points
    x, y = pattern_shape
    objp = np.zeros((x * y, 3), np.float32)
    grid = np.mgrid[0:x, 0:y].T.reshape(-1, 2) * side_length
    objp[:, 0] = grid[:, 0]
    objp[:, 2] = grid[:, 1]

    imgp = cv2.cornerSubPix(image,
                            corners,
                            (11, 11),
                            (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # Solve perspective from n points
    success, rvec, tvec = cv2.solvePnP(np.array(objp),
                                       np.array(imgp),
                                       camera_intrinsic_matrix,
                                       distortion_coefficients,
                                       flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        R = cv2.Rodrigues(rvec)[0]
        extrinsic_matrix = np.hstack((R, tvec))
        projection_matrix = np.dot(camera_intrinsic_matrix, extrinsic_matrix)
        return rvec, tvec, extrinsic_matrix, projection_matrix

    return None


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
            # If the current point matches the previous key frame's point then they reference the same feature
            prior_point = track.getCoordinate(prev_keyframe_ID)

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


def triangulatePoints(tracks, projections):
    """
    Generates the new 3D points from the popped_tracks and poses as well as keeping track of how the points and
    frames link together

    :param tracks: The tracks that will not be updated again
    :param projections: The poses of the frames so far
    """
    for track in tracks:
        frame_ID1, frame_ID2, feature, correspondent = track.getTriangulationData()
        projection1 = projections[frame_ID1]
        projection2 = projections[frame_ID2]

        point = cv2.triangulatePoints(projection1, projection2, feature, correspondent).T
        point = point[:, :3] / point[:, -1, np.newaxis]
        track.setPoint(point)


def managePoints(tracks):
    """
    Generates the correspondences between image coordinates, frame extrinsics and points

    :param tracks: The relevant feature tracks
    :return: 3D point array,
            2D image coordinate array,
            array of point indices corresponding to each coordinate
            Array of frame indices corresponding to each coordinate
    """
    point_index = 0
    points = []
    coordinates = []
    point_indices = []
    frame_indices = []

    for track in tracks:
        point = track.getPoint()
        points.append(point)

        for frame_index, coordinate in track.getCoordinates().items():
            coordinates.append(coordinate)
            point_indices.append(point_index)
            frame_indices.append(frame_index)

        point_index += 1

    return points, coordinates, frame_indices, point_indices


def process(video, path, lk_params, feature_params, flann_params):
    """
    Takes a video of a food item and returns the 3D mesh of the food item

    :param video: The video to be converted to a 3D mesh
    :param path: The path to save images to
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
    prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
    has_chessboard, corners = cv2.findChessboardCorners(prev_frame_grey, (4, 3))
    while not has_chessboard:
        _, start_frame = cap.read()
        prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
        has_chessboard, corners = cv2.findChessboardCorners(prev_frame_grey, (4, 3))

    # Initialise keyframe tracking
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Initialise feature tracking
    prev_orb_points, prev_orb_descriptors = orb.detectAndCompute(prev_frame_grey, None)

    # Initialise pose estimation
    frame_corners = [corners]
    frames = [prev_frame_grey]
    extrinsic_matrices = []
    projections = []
    rvecs = []
    tvecs = []

    # Initialise point tracking
    tracks = []
    popped_tracks = []
    prev_keyframe_ID = 0
    keyframe_ID = 1

    toc = time.time()

    print("Initialisation complete.")
    print(toc - tic, "seconds.\n")

    print("Finding points...")

    tic = time.time()

    # Processing loop
    success, frame = cap.read()

    while success:
        frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)

        is_keyframe, prev_frame_grey, prev_frame_points, accumulative_error = keyframeTracking(frame_grey,
                                                                                               prev_frame_grey,
                                                                                               prev_frame_points,
                                                                                               accumulative_error,
                                                                                               lk_params,
                                                                                               feature_params,
                                                                                               threshold=0.1)

        if is_keyframe:
            # Find chessboard corners
            has_chessboard, corners = cv2.findChessboardCorners(frame_grey, (4, 3))

            if has_chessboard:
                # Append frame corners
                frame_corners.append(corners)
                frames.append(frame_grey)

                # Calculate matches
                print("\nFinding matches", end="...")
                prev_matches, curr_matches, prev_orb_points, prev_orb_descriptors = featureTracking(frame_grey,
                                                                                                    prev_orb_points,
                                                                                                    prev_orb_descriptors,
                                                                                                    orb,
                                                                                                    flann_params)
                print("found", len(prev_matches))

                # Manage tracks
                print("Grouping points", end="...")
                new_popped_tracks, tracks = pointTracking(tracks,
                                                          prev_keyframe_ID,
                                                          prev_matches,
                                                          keyframe_ID,
                                                          curr_matches)
                popped_tracks += new_popped_tracks
                print(len(popped_tracks) + len(tracks), "potential points")

                # if new_popped_tracks:
                #     # Triangulating points
                #     print("Triangulating points", end="...")
                #     triangulatePoints(popped_tracks, projections)
                #     print(len(popped_tracks), "triangulated")
                #
                #     # Adjusting frame parameters and points
                #     print("Adjusting frames and points", end="...")
                #     points, points_2d, frame_indices, point_indices = managePoints(popped_tracks)
                #
                #     adjusted_points, extrinsic_matrices = bundleAdjuster.adjustPoints(np.array(extrinsic_matrices),
                #                                                                       intrinsic_matrix,
                #                                                                       np.array(points),
                #                                                                       np.array(points_2d),
                #                                                                       np.array(frame_indices),
                #                                                                       np.array(point_indices))
                #
                #     projections = list(np.einsum("ij,...jk", intrinsic_matrix, np.array(extrinsic_matrices)[:, :3, :]))
                #     print("done")

                # Update variables
                # prev_extrinsic = extrinsic_matrices[-1]  # previous frame was last extrinsic frame
                prev_keyframe_ID = keyframe_ID
                keyframe_ID += 1

        success, frame = cap.read()

    # Add the remaining tracks which are implicitly popped
    popped_tracks += tracks

    # Calibration
    print("\nCalibrating camera", end="...")
    intrinsic_matrix, distortion_coefficients = calibrate(frame_corners, prev_frame_grey.shape[::-1], (4, 3))
    print("done", end="\n\n")

    # Pose estimation
    print("Estimating frame positions...")
    count = 0
    for corners, frame in zip(frame_corners, frames):
        rvec, tvec, extrinsic_matrix, projection_matrix = poseEstimation(corners,
                                                                         frame,
                                                                         (4, 3),
                                                                         2,
                                                                         intrinsic_matrix,
                                                                         distortion_coefficients)

        rvecs.append(rvec)
        tvecs.append(tvec)
        extrinsic_matrices.append(extrinsic_matrix)
        projections.append(projection_matrix)
        print("Frame", count)
        count += 1
    print("Done", end="\n\n")

    # # Include the points in the tracks not popped at the end
    # print("\nTriangulating all points", end="...")
    # triangulatePoints(popped_tracks, projections)
    # print("done")
    #
    # toc = time.time()
    #
    # print(len(extrinsic_matrices), "frames used")
    # print(toc - tic, "seconds\n")
    #
    # print("adjusting points...")
    # tic = time.time()
    #
    # points, points_2d, frame_indices, point_indices = managePoints(popped_tracks)
    #
    # adjusted_points, adjusted_positions = bundleAdjuster.adjustPoints(np.array(extrinsic_matrices),
    #                                                                   intrinsic_matrix,
    #                                                                   np.array(points),
    #                                                                   np.array(points_2d),
    #                                                                   np.array(frame_indices),
    #                                                                   np.array(point_indices))
    #
    # toc = time.time()
    # print("adjustment complete.")
    # print(len(adjusted_points), "points found")
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
    # print(toc - tic)
