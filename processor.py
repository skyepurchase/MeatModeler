import time

import cv2
import numpy as np
import pandas as pd
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


def poseEstimation(left_frame_points, right_frame_points, left_frame_extrinsic_matrix, camera_intrinsic_matrix):
    """
    Takes the matches between two frames and the transformation between origin and left frame coordinates and finds
    the transformation between origin and right frame coordinates

    :param left_frame_points: Undistorted matched points from the left frame
    :param right_frame_points: Undistorted matched points from the right frame
    :param left_frame_extrinsic_matrix: 4x4 matrix converting origin coordinates to left frame coordinates
    :param camera_intrinsic_matrix: The intrinsic matrix of the camera
    :return: The used left points,
            The used right points,
            The corresponding 3D points,
            The new previous pose matrix
    """
    # Find essential matrix and inliers
    essential_matrix, mask_E = cv2.findEssentialMat(left_frame_points,
                                                    right_frame_points,
                                                    camera_intrinsic_matrix)

    # Use the essential matrix and inliers to find the pose and new inliers
    _, R_left_to_right, t_left_to_right, mask_RP = cv2.recoverPose(essential_matrix,
                                                                   left_frame_points,
                                                                   right_frame_points,
                                                                   camera_intrinsic_matrix,
                                                                   mask=mask_E)

    # Create the 4x3 pose matrix from rotation and translation
    left_to_right_extrinsic_matrix = np.hstack([R_left_to_right, t_left_to_right])

    # Convert to homogeneous 4x4 transformation matrix
    left_to_right_extrinsic_matrix = np.vstack((left_to_right_extrinsic_matrix, np.array([0, 0, 0, 1])))

    # Take world coordinates to left frame then to right frame
    right_frame_extrinsic_matrix = np.matmul(left_to_right_extrinsic_matrix, left_frame_extrinsic_matrix)

    # Projection from world coordinates to right frame image coordinates
    projection_matrix = np.dot(camera_intrinsic_matrix, right_frame_extrinsic_matrix[:3])

    # Usable points
    usable_left_points = left_frame_points[mask_RP[:, 0] == 1]
    usable_right_points = right_frame_points[mask_RP[:, 0] == 1]

    return usable_left_points, usable_right_points, right_frame_extrinsic_matrix, projection_matrix


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


def managePoints(popped_tracks, projections, point_ID, points_2d, frame_indices, point_indices):
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
        new_points = new_points[:, :3] / new_points[:, 3][:, None]

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
    left_frame_extrinsic_matrix = np.eye(4, 4)  # The first keyframe is at origin and left of next frame

    # But needs to be placed into world coordinates
    original_projection = np.dot(intrinsic_matrix, left_frame_extrinsic_matrix[:3])

    projections = [original_projection]  # The first keyframe is added
    extrinsic_matrices = [left_frame_extrinsic_matrix]

    # Initialise point tracking
    tracks = []
    popped_tracks = []
    prev_keyframe_ID = 0
    keyframe_ID = 1

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
        # Whether the start frame is rejoined
        has_joined = False

        frame = undistortFrame(frame, intrinsic_matrix, distortion_coefficients)
        frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)

        is_keyframe, prev_frame_grey, prev_frame_points, accumulative_error = keyframeTracking(frame_grey,
                                                                                               prev_frame_grey,
                                                                                               prev_frame_points,
                                                                                               accumulative_error,
                                                                                               lk_params,
                                                                                               feature_params,
                                                                                               threshold=0.1)

        if is_keyframe:
            # Calculate matches
            L_matches, R_matches, prev_orb_points, prev_orb_descriptors = featureTracking(frame_grey,
                                                                                          prev_orb_points,
                                                                                          prev_orb_descriptors,
                                                                                          orb,
                                                                                          flann_params)

            start_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)

            # Calculate similarity
            start_L_matches, start_R_matches, features, _ = featureTracking(start_frame_grey,
                                                                            prev_orb_points,
                                                                            prev_orb_descriptors,
                                                                            orb,
                                                                            flann_params)

            # If there is a greater similarity to the start frame than to the previous frame the loop has closed
            if (len(start_L_matches) / len(features)) >= (len(L_matches) / len(prev_orb_points)) and keyframe_ID > 1:
                L_matches = start_L_matches
                R_matches = start_R_matches
                keyframe_ID = 0
                has_joined = True

            # Pose estimation
            L_points, R_points, right_frame_extrinsic_matrix, projection = poseEstimation(L_matches,
                                                                                          R_matches,
                                                                                          left_frame_extrinsic_matrix,
                                                                                          intrinsic_matrix)
            if has_joined:
                origin_error = right_frame_extrinsic_matrix  # The origin to right should return to origin
            else:
                projections.append(projection)  # Do not add origin twice
                extrinsic_matrices.append(right_frame_extrinsic_matrix)

            # Manage tracks
            new_popped_tracks, tracks = pointTracking(tracks,
                                                      prev_keyframe_ID,
                                                      L_points,
                                                      keyframe_ID,
                                                      R_points)
            popped_tracks += new_popped_tracks

            # Update variables
            left_frame_extrinsic_matrix = right_frame_extrinsic_matrix  # Right keyframe now becomes the left keyframe
            prev_keyframe_ID = keyframe_ID
            keyframe_ID += 1

        success, frame = cap.read()

        # If loop has occurred don't keep processing
        if has_joined:
            success = False

    # Add the remaining tracks which are implicitly popped
    popped_tracks += tracks

    # Include the points in the tracks not popped at the end
    points, point_ID, points_2d, frame_indices, point_indices = managePoints(popped_tracks,
                                                                             projections,
                                                                             point_ID,
                                                                             points_2d,
                                                                             frame_indices,
                                                                             point_indices)

    toc = time.time()
    print(len(points), "points found.")
    print(toc - tic, "seconds.\n")

    print("adjusting points...")
    tic = time.time()

    adjusted_points, adjusted_positions = bundleAdjuster.adjustPoints(np.array(extrinsic_matrices),
                                                                      intrinsic_matrix,
                                                                      points,
                                                                      np.array(points_2d),
                                                                      np.array(frame_indices),
                                                                      np.array(point_indices))

    toc = time.time()
    print("adjustment complete.")
    print(toc - tic, "seconds.\n")

    print("Saving point cloud...")
    tic = time.time()

    filename = path + "Cloud.ply"
    cloud = PyntCloud(pd.DataFrame(
        data=adjusted_points,
        columns=['x', 'y', 'z']
    ))
    cloud.to_file(filename)
