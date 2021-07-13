import cv2
import numpy as np
import bundleAdjuster
from track import Track


# Greyscale frame in
# Frame with increased contrast out
# Uses CLAHE to normalise the luminosity
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


# Chess images in
# Intrinsic matrix and distortion coefficients out
# Standard for all frames
def calibrate(images):
    """
    Takes specific chess board images and calibrates the camera appropriately

    :param images: A list of different images of a known chessboard
    :return: The intrinsic property matrix,
            The distortion coefficients
    """
    # Prepare chessboard 3D points
    # TODO: allow different numbers of corners
    objp = np.zeros((7 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    # Arrays to store object and image points from all images
    obj_points = []
    img_points = []

    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        success, corners = cv2.findChessboardCorners(gray, (7, 7), None)

        # If found, add object points, image points
        if success:
            obj_points.append(objp)
            img_points.append(corners)

    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    success, matrix, distortion, _, _ = cv2.calibrateCamera(obj_points,
                                                            img_points,
                                                            gray.shape[::-1],
                                                            None,
                                                            None)

    if success:
        return matrix, distortion

    return None


# Frame, camera_matrix, and distortion coefficients in
# Undistorted frame out
# Do not need to undistort points later on
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


# Greyscale frame and feature points in
# Greyscale frame and feature points out
# The frames have distortion removed so feature points are undistorted
def keyframeTracking(frame_grey, prev_frame_grey, prev_frame_points, accumulated_error, lk_params, feature_params,
                     threshold=0.3):
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
        else:
            return False, prev_frame_grey, prev_frame_points, accumulated_error


# Greyscale frame, feature points, and descriptors in
# Matched points, feature points, and descriptors out
# Points have distortion removed as the frames are undistorted
def featureTracking(new_keyframe, prev_orb_points, prev_orb_descriptors, orb, flann_params, old_frame):
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


# Point matches and previous frame position in
# Used points and frame position out
# Points are still undistorted
def poseEstimation(left_frame_points, right_frame_points, prev_pose, camera_matrix):
    """
    Takes the matches between two frames and finds the rotation and translation of the second frame

    :param left_frame_points: Undistorted matched points from the left frame
    :param right_frame_points: Undistorted matched points from the right frame
    :param prev_pose: The pose of the left frame in relation to the original frame
    :param camera_matrix: The intrinsic matrix of the camera
    :return: The used left points,
            The used right points,
            The corresponding 3D points,
            The new previous pose matrix
    """
    # Find essential matrix and inliers
    # Use focal length as 1 and centre as (0, 0) because frames and points already undistorted
    essential, mask_E = cv2.findEssentialMat(left_frame_points,
                                             right_frame_points,
                                             camera_matrix)

    # Use the essential matrix and inliers to find the pose and new inliers
    points, R, t, mask_RP = cv2.recoverPose(essential,
                                            left_frame_points,
                                            right_frame_points,
                                            camera_matrix,
                                            mask=mask_E)

    # Create the 3x4 pose matrices
    pose_transform = np.vstack([prev_pose, np.array([0, 0, 0, 1])])
    Pose2 = np.matmul(np.hstack([R, t]), pose_transform)

    # Usable points
    usable_left_points = left_frame_points[mask_RP[:, 0] == 1]
    usable_right_points = right_frame_points[mask_RP[:, 0] == 1]

    return usable_left_points, usable_right_points, Pose2


# Tracks, frame IDs, frame positions, and matches in
# Tracks (With frame positions and undistorted points) to process and to keep out
# Points are still undistorted
def pointTracking(tracks, prev_keyframe_ID, prev_keyframe_pose, feature_points, keyframe_ID, keyframe_pose,
                  correspondents):
    """
    Checks through the current tracks and updates them based on the provided matches

    :param tracks: Current tracks
    :param prev_keyframe_ID: The identity number of the previous keyframe
    :param prev_keyframe_pose: The absolute pose of the previous keyframe
    :param feature_points: The feature point matches from the previous keyframe
    :param keyframe_ID: The identity number of the current keyframe
    :param keyframe_pose: The absolute pose of the current keyframe
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
                track.update(keyframe_ID, keyframe_pose, correspondent)
                is_new_track = False
                break

        # Feature was not found elsewhere
        if is_new_track:
            new_track = Track(prev_keyframe_ID,
                              prev_keyframe_pose,
                              feature_point,
                              keyframe_ID,
                              keyframe_pose,
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


# Frame positions and undistorted points in
# Single 3D point out
# 3D point is in world coordinates based on poses
def triangulation(first_pose, last_pose, features):
    """
    Using the furthest apart frames calculates the 3D position of a given point

    :param first_pose: The absolute position of the first frame
    :param last_pose: The absolute position of the last frame
    :param features: The list of feature image coordinates
    :return: The resulting 3D point
    """
    # Use the poses to find the homogeneous 3D points
    homogeneous_point = cv2.triangulatePoints(first_pose,
                                              last_pose,
                                              np.array(features[0]),
                                              np.array(features[-1])).T

    # Normalise homogeneous (w=1)
    norm_point = homogeneous_point / homogeneous_point[:, -1][:, None]
    norm_point = norm_point[0][:3]

    return norm_point


# Video, camera and processing parameters in
# 3D point cloud out
# The main processing loop
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
    orb = cv2.ORB_create(nfeatures=20000)

    cap = cv2.VideoCapture(video)

    # Retrieve first frame
    _, start_frame = cap.read()
    start_frame = undistortFrame(start_frame, intrinsic_matrix, distortion_coefficients)
    count = 1

    # Initialise keyframe tracking
    prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Initialise feature tracking
    prev_orb_points, prev_orb_descriptors = orb.detectAndCompute(prev_frame_grey, None)
    old_frame = prev_frame_grey

    # Initialise pose estimation
    prev_pose = np.hstack([np.eye(3, 3), np.zeros((3, 1))])

    # Initialise point tracking
    tracks = []
    prev_keyframe_ID = 0
    keyframe_ID = 1

    # Initialise bundling
    frame_projections = [prev_pose]
    points = []
    observations = []
    frame_indices = []
    point_indices = []
    point_ID = 0

    # TODO: remove
    filename = path + "Raw\\Image0.jpg"
    cv2.imwrite(filename, start_frame)

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
                                                                                          flann_params,
                                                                                          old_frame)
            old_frame = prev_frame_grey

            # Pose estimation
            L_points, R_points, pose = poseEstimation(L_matches,
                                                      R_matches,
                                                      prev_pose,
                                                      intrinsic_matrix)
            frame_projections.append(pose)

            # Update tracks
            popped_tracks, tracks = pointTracking(tracks,
                                                  prev_keyframe_ID,
                                                  prev_pose,
                                                  L_points,
                                                  keyframe_ID,
                                                  pose,
                                                  R_points)

            # Triangulation
            for track in popped_tracks:
                first_frame_ID, first_pose, last_frame_ID, last_pose, features = track.getTriangulationData()

                # Get the 3D point and store
                point = triangulation(first_pose, last_pose, features)
                points.append(point)

                # Relate the features to a frame and point
                for i in range(first_frame_ID, last_frame_ID + 1):
                    observations.append(features[i - first_frame_ID])
                    point_indices.append(point_ID)
                    frame_indices.append(i)

                point_ID += 1

            # Update variables
            prev_pose = pose
            prev_keyframe_ID = keyframe_ID
            keyframe_ID += 1

            # TODO: remove
            filename = path + "Raw\\Image" + str(count) + ".jpg"
            cv2.imwrite(filename, frame)
            count += 1
            # print("Number of new points:", len(prev_orb_points))
            # print("Number of matches:", len(L_matches))
            # print("Number of which usable:", len(L_points))
            # print("Number of tracks popped:", len(popped_tracks))
            # print()

        success, frame = cap.read()

    points = np.array(points)
    projections = np.array(frame_projections)

    adjusted_points = bundleAdjuster.bundleAdjustment(projections,
                                                      intrinsic_matrix,
                                                      points,
                                                      np.array(observations),
                                                      np.array(frame_indices),
                                                      np.array(point_indices))
