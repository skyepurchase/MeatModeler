import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pyntcloud import PyntCloud
from PIL import Image, ExifTags

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


def videoCalibrate(video, feature_params, lk_params, corner_dims=(7, 7)):
    """
    Takes a video input of a chessboard pattern and returns the calibration parameters of the camera

    :param video: 360* video of chessboard pattern
    :param feature_params: Parameters for openCV goodFeaturesToTrack
    :param lk_params: Parameters for keyFrameTracking function
    :param corner_dims: The dimension of chessboard corners
    :return: intrinsic camera matrix,
            distortion coefficients
    """
    images = []

    # Retrieve first frame
    _, start_frame = video.read()

    # Initialise keyframe tracking
    prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Processing loop
    success, frame = video.read()

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
            images.append(frame)

        success, frame = video.read()

    return calibrate(images, corner_dims)


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


def intrinsicFromEXIF(image):
    """
    Generates an intrinsic matrix based on the EXIF data of an image.
    A number of assumptions are used and no distortion coefficients can be generated.
    The intrinsic matrix will now be in real world coordinates.

    :param image: The path to an image taken from the used phone
    :return: The intrinsic matrix
    """

    # Extract EXIF data
    tags = {}
    with open(image, 'rb') as image_file:
        img = Image.open(image_file)
        if hasattr(img, '_getexif'):
            exif_info = img._getexif()
            if exif_info is not None:
                for tag, value in exif_info.items():
                    tags[ExifTags.TAGS.get(tag, tag)] = value

    # Extract focal length
    focal_length = tags.get('FocalLength', (0, 1))

    # Extract resolution
    img_width = tags.get('XResolution', 0)
    img_height = tags.get('YResolution', 0)
    if img_width < img_height:
        img_width, img_height = img_height, img_width

    # Extract DPI resolutions
    f_planeN_X, f_planeD_X = tags.get('FocalPlaneXResolution', (0, 1))
    f_planeN_Y, f_planeD_Y = tags.get('FocalPlaneYResolution', (0, 1))
    X_resolution = f_planeN_X / f_planeD_X
    Y_resolution = f_planeN_Y / f_planeD_Y

    # focal length is in mm, resolution in px / "
    # fx and fy in px
    fx = focal_length * X_resolution / 25.4
    fy = focal_length * Y_resolution / 25.4

    # Assume the principle point is at the centre
    cx, cy = (img_width / 2, img_height / 2)

    return np.hstack((np.array([fx, 0, cx]),
                      np.array([0, fy, cy]),
                      np.array([0, 0, 1])))


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


def featureMatching(new_features, new_descriptors, all_features, all_descriptors, flann_params, threshold=0.75):
    """
    Finds which features in two keyframes match

    :param new_features: The features from the new keyframe
    :param new_descriptors: The feature descriptions from the new keyframe
    :param all_features: The previous features from all preceding keyframes
    :param all_descriptors: The previous feature descriptions from all preceding keyframes
    :param flann_params: Parameters to tune FLANN matcher
    :param threshold: Ratio threshold for FLANN matches
    :return: A dictionary of the matches for each preceding keyframe
    """
    # Get new points and descriptors
    all_matches = {}

    for frame_ID, (points, descriptors) in enumerate(zip(all_features, all_descriptors)):
        # FLANN based approach to find matches
        flann = cv2.FlannBasedMatcher(flann_params, {})
        matches = flann.knnMatch(descriptors, new_descriptors, k=2)

        # Find which points can be considered new
        good_matches = [match[0] for match in matches if
                        len(match) == 2  # Sometimes matches only includes one side not both
                        and match[0].distance < threshold * match[1].distance]
        print("Found", len(good_matches), "matches with keyframe", frame_ID)

        left_matches = np.array([points[m.queryIdx].pt for m in good_matches])
        right_matches = np.array([new_features[m.trainIdx].pt for m in good_matches])
        new_matches = np.hstack((left_matches, right_matches))
        all_matches[frame_ID] = new_matches

    return all_matches


def initialPoseEstimation(points, camera_intrinsic_matrix):
    """
    Takes the matches between two frames and the transformation between origin and left frame coordinates and finds
    the transformation between origin and right frame coordinates

    :param points: Nx4 array of image point matches
    :param camera_intrinsic_matrix: The intrinsic matrix of the camera
    :return: The used point matches (or if estimating based on previous points, a success value),
            rotation vector,
            translation vector
    """
    left_points = points[:, :2]
    right_points = points[:, 2:]

    # Find essential matrix and inliers
    E, mask_E = cv2.findEssentialMat(left_points,
                                     right_points,
                                     camera_intrinsic_matrix)

    # Use the essential matrix and inliers to find the pose and new inliers
    _, R, t, mask_RP, points = cv2.recoverPose(E,
                                               left_points,
                                               right_points,
                                               camera_intrinsic_matrix,
                                               distanceThresh=10,
                                               mask=mask_E)

    new_points = points.T
    new_points = new_points[:, :3] / new_points[:, -1][:, None]

    # Usable points
    usable_left_points = left_points[mask_RP[:, 0] == 1]

    if len(usable_left_points) < 8:  # If less than 8 are usable then this is very unreliable
        return None, None, None

    usable_right_points = right_points[mask_RP[:, 0] == 1]
    usable_points = np.hstack((usable_left_points, usable_right_points))
    usable_new_points = new_points[mask_RP[:, 0] == 1]

    return usable_points, R, t, usable_new_points


def poseEstimation(all_matches, tracks, camera_intrinsic_matrix):
    """
    Estimates the pose of a new frame based on previously triangulated 3D points

    :param all_matches: A dictionary of all the matches with previous frames (from featureMatching)
    :param tracks: A dictionary of all the tracks present in each frame
    :param camera_intrinsic_matrix: The intrinic matrix of the camera used
    :return: Whether the estimation was successful
            The rotation vector,
            The translation vector
    """
    obj_points = []
    img_points = []
    for prev_keyframe_ID, matches in all_matches.items():
        potential_tracks = tracks.get(prev_keyframe_ID)

        for track in potential_tracks:
            potential_coordinate = track.getCoordinate(prev_keyframe_ID)

            if potential_coordinate in matches[:, :2]:
                obj_points.append(track.getPoint())
                img_points.append(potential_coordinate)

    success, rvec, tvec = cv2.solvePnP(np.array(obj_points),
                                       np.array(img_points),
                                       camera_intrinsic_matrix,
                                       np.zeros((4, 1)),
                                       flags=0)

    return success, rvec, tvec


def triangulatePoints(matched_points, projection1_params, projection2_params, camera_intrinsic_matrix):
    """
    Takes a Nx4 array of matched points and projection matrices returning the corresponding 3D points

    :param matched_points: Nx4 array of image points [projection1, projection2]
    :param projection1_params: Either 3x4 matrix or tuple of rotation vector and translation vector if vector is True
    :param projection2_params: Same as above. Both project world coordinates into image corresponding image coordinates
    :param camera_intrinsic_matrix: The intrinsic properties of the camera used
    :return: Nx3 array corresponding to the input points
    """
    # Convert to matrix form
    rvec1, tvec1 = projection1_params
    rvec2, tvec2 = projection2_params

    rotation1, _ = cv2.Rodrigues(rvec1)
    rotation2, _ = cv2.Rodrigues(rvec2)

    extrinsic1 = np.hstack((rotation1, tvec1))
    extrinsic2 = np.hstack((rotation2, tvec2))

    projection1 = np.dot(camera_intrinsic_matrix, extrinsic1)
    projection2 = np.dot(camera_intrinsic_matrix, extrinsic2)

    new_points = cv2.triangulatePoints(projection1,
                                       projection2,
                                       matched_points[:, :2].T,
                                       matched_points[:, 2:].T).T

    new_points = new_points[:, :3] / new_points[:, -1][:, None]
    return new_points


def pointTracking(tracks, matches, points, prev_keyframe_ID, keyframe_ID):
    """
    Checks through the current tracks and updates them based on the provided matches

    :param tracks: Tracks of points visible in the previous keyframe
    :param matches: Nx4 array of the matches between the keyframe and previous keyframe
    :param points: Nx3 corresponding 3D points
    :param prev_keyframe_ID: The identity number of the previous keyframe
    :param keyframe_ID: The identity number of the current keyframe
    :return: The tracks for the previous keyframe,
            The tracks for the current keyframe,
            The new tracks created
    """
    new_tracks = []
    new_frame_tracks = []

    # For each match check if this feature already exists
    for feature_point, correspondent, point in zip(matches[:, :2], matches[:, 2:], points):
        is_new_track = True

        for track in tracks:
            # If the current point matches the track's last point then they reference the same feature
            prior_point = track.getCoordinate(prev_keyframe_ID)

            # So update the track
            if feature_point[0] == prior_point[0] and feature_point[1] == prior_point[1]:
                track.update(keyframe_ID, correspondent, point)
                new_frame_tracks.append(track)
                is_new_track = False
                break

        # Feature was not found elsewhere
        if is_new_track:
            new_track = Track(prev_keyframe_ID,
                              feature_point,
                              keyframe_ID,
                              correspondent,
                              point)
            new_tracks.append(new_track)
            new_frame_tracks.append(new_track)

    # Add new tracks
    tracks += new_tracks

    return tracks, new_frame_tracks, new_tracks


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

    orb = cv2.ORB_create(nfeatures=2000)

    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Retrieve first frame
    _, start_frame = cap.read()

    # Initialise keyframe tracking
    prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
    prev_frame_points = cv2.goodFeaturesToTrack(prev_frame_grey,
                                                mask=None,
                                                **feature_params)
    accumulative_error = 0

    # Initialise feature matching
    features, descriptors = orb.detectAndCompute(prev_frame_grey, None)
    all_features = [features]
    all_descriptors = [descriptors]

    # Initialise pose estimation
    frame_tracks = {}

    # Initialise triangulation

    # Initialise point tracking
    tracks = []
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
            # Detect features and compute descriptors
            print("Detecting features in keyframe", keyframe_ID, end="...")

            features, descriptors = orb.detectAndCompute(frame_grey, None)

            print("found", len(features))

            # Calculate matches
            all_matches = featureMatching(features,
                                          descriptors,
                                          all_features,
                                          all_descriptors,
                                          flann_params)

            # Update all features and descriptors
            all_features.append(features)
            all_descriptors.append(descriptors)

            # Pose estimation

            # Pose Assumption
            # Use openCV recoverPose to get a base assumption of the relative location of the first two frames
            if keyframe_ID == 1:
                print("Finding inliers between first keyframes", end="...")

                matches = all_matches[0]  # Get matches between this frame (frame 1) and the first frame (frame 0)
                usable_matches, R, t, new_points = initialPoseEstimation(matches,
                                                                         intrinsic_matrix)

                print("found", len(usable_matches[:, 0]))

                # Create tracks for each of the new points
                prev_tracks, new_frame_tracks, new_tracks = pointTracking([],
                                                                          matches,
                                                                          new_points,
                                                                          0,
                                                                          1)

                frame_tracks[0] = prev_tracks
                frame_tracks[1] = new_frame_tracks
                tracks += new_tracks
            else:
                current_frame_tracks = []
                for prev_keyframe_ID, matches in all_matches.items():
                    print("Finding points with frame", prev_keyframe_ID, end="...")
                    # Find the relative positions and triangulated points
                    usable_matches, R, t, new_points = initialPoseEstimation(matches,
                                                                             intrinsic_matrix)

                    print("found", len(new_points))

                    # Group the newly triangulated points into tracks
                    prev_tracks, new_frame_tracks, new_tracks = pointTracking(frame_tracks[prev_keyframe_ID],
                                                                              usable_matches,
                                                                              new_points,
                                                                              prev_keyframe_ID,
                                                                              keyframe_ID)

                    frame_tracks[prev_keyframe_ID] = prev_tracks
                    current_frame_tracks += new_frame_tracks
                    tracks += new_tracks

                print(len(tracks), "potential points found.")
                frame_tracks[keyframe_ID] = current_frame_tracks

            # Update variables
            keyframe_ID += 1

        success, frame = cap.read()

    toc = time.time()

    print(len(tracks), "points found.")
    # print(len(extrinsic_vectors), "frames used.")
    print(toc - tic, "seconds.\n")

    points = []
    point_indices = []
    frame_indices = []
    points_2d = []
    point_ID = 0

    for track in tracks:
        for frame_ID, coordinate in track.getCoordinates().items():
            points_2d.append(coordinate)
            frame_indices.append(frame_ID)
            point_indices.append(point_ID)

        points.append(track.getFinalPoint())
        point_ID += 1

    points = np.array(points)

    print("adjusting points...")

    tic = time.time()

    # adjusted_points, adjusted_positions = bundleAdjuster.adjustPoints(frame_parameters,
    #                                                                   intrinsic_matrix,
    #                                                                   np.array(points),
    #                                                                   np.array(points_2d),
    #                                                                   np.array(frame_indices),
    #                                                                   np.array(point_indices))

    toc = time.time()

    print("adjustment complete.")
    print(toc - tic, "seconds.\n")

    print("Saving point cloud...")

    tic = time.time()

    filename = path + "Cloud.ply"
    cloud = PyntCloud(pd.DataFrame(
        data=points,
        columns=['x', 'y', 'z']
    ))
    cloud.to_file(filename)

    toc = time.time()

    print(toc - tic, "seconds.\n")
