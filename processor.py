import cv2
import numpy as np
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


class Processor:
    def __init__(self, images):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.flann_params = dict(algorithm=6,
                                 table_number=6,
                                 key_size=12,
                                 multi_probe_level=2)

        self.intrinsic, self.distortion = self.calibrate(images)

        self.prev_frame_grey = None
        self.prev_frame_points = None
        self.prev_keyframe_grey = None
        self.acc_error = 0

        self.current_orb_points = None
        self.current_orb_descriptors = None
        self.point_vector = []
        self.orb = cv2.ORB_create(nfeatures=2000)

        self.tracks = []

        self.extrinsic_properties = {}

        # Debugging stuff
        self.color = np.random.randint(0, 255, (100, 3))
        self.display = False
        self.mask = None
        self.count = 1

    def calibrate(self, images):
        # Prepare chessboard 3D points
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

    def process(self, video, display=False):
        """
        Takes a video of a food item and returns the 3D mesh of the food item
        :param video: The video to be converted to a 3D mesh
        :param display: Whether the process should be displayed
        :return: A 3D mesh
        """
        # TODO: utilise ORB rather than goodFeaturesToTrack
        self.display = display

        cap = cv2.VideoCapture(video)

        # Extract features from the start frame
        _, start_frame = cap.read()
        self.prev_frame_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
        self.prev_keyframe_grey = self.prev_frame_grey
        self.prev_frame_points = cv2.goodFeaturesToTrack(self.prev_frame_grey,
                                                         mask=None,
                                                         **self.feature_params)
        self.current_orb_points, self.current_orb_descriptors = self.orb.detectAndCompute(self.prev_frame_grey, None)

        # Will be removed
        cv2.imwrite("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Raw\\Image0.jpg", start_frame)
        self.mask = np.zeros_like(start_frame)

        # Processing loop
        success, frame = cap.read()

        while success:
            frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)

            if self.isKeyframe(frame_grey):
                # Calculate matches
                matches = self.calculateMatchedPoints(frame_grey)

                # Pose estimation
                R, t = self.findRotationAndTranslation(matches)

                # Update tracks
                self.manageTracks(frame_grey, matches)

                for track in self.tracks:
                    if track.wasUpdated():
                        self.tracks.remove(track)

                # Will be removed later
                self.mask = np.zeros_like(frame)
                filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Raw\\Image" + str(
                    self.count) + ".jpg"
                cv2.imwrite(filename, frame)
                self.count += 1

            success, frame = cap.read()

    def isKeyframe(self, frame):
        """
        Determines whether a given frame is a keyframe for further analysis
        :param frame: The frame to be analysed
        :return: A boolean value on whether the frame was a keyframe
        """
        # Compare the last key frame to current key frame
        p, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_grey,
                                              frame,
                                              self.prev_frame_points,
                                              None,
                                              **self.lk_params)

        # Keep only matching points
        if p is not None:
            good_new = p[st == 1]
            good_prev = self.prev_frame_points[st == 1]

            # Will be removed later
            if self.display:
                for i, (new, old) in enumerate(zip(good_new, good_prev)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

                img = cv2.add(frame, self.mask)
                cv2.imshow("Tracking", img)
                key = cv2.waitKey() & 0xff
                if key == 27:
                    self.display = False

            # Update previous data
            self.prev_frame_grey = frame
            self.prev_frame_points = good_new.reshape(-1, 1, 2)

            # If possible increase the accumulative error between frames
            if err is not None:
                self.acc_error += np.average(err)

            # Current frame has deviated enough to be considered a key frame
            if self.acc_error > 0.3 * frame.shape[1]:
                self.acc_error = 0

                # Recalculate points for new keyframe
                self.prev_frame_points = cv2.goodFeaturesToTrack(self.prev_frame_grey,
                                                                 mask=None,
                                                                 **self.feature_params)
                return True
            else:
                return False

    def calculateMatchedPoints(self, keyframe):
        """
        Finds which features in two keyframes match
        :param keyframe: The keyframe to compare to the previous keyframe
        :return: List of keypoints
        """
        # Get new points and descriptors
        new_points, new_descriptors = self.orb.detectAndCompute(keyframe, None)

        # FLANN based approach to find matches
        flann = cv2.FlannBasedMatcher(self.flann_params, {})
        matches = flann.knnMatch(self.current_orb_descriptors, new_descriptors, k=2)

        # Find which points can be considered new
        # TODO: vectorise following calculations
        good_matches = [match[0] for match in matches if
                        len(match) == 2 and match[0].distance < 0.8 * match[1].distance]
        point_matches = np.array(
            [[self.current_orb_points[m.queryIdx].pt, new_points[m.trainIdx].pt] for m in good_matches])

        # Update the keyframe points
        self.current_orb_points, self.current_orb_descriptors = new_points, new_descriptors

        return point_matches

    def manageTracks(self, keyframe, matches):
        """
        Checks through the current tracks and updates them based on the matches.
        If there are new features a new track is made.
        If a track is not updated it is tagged.
        :param keyframe: The keyframe to be analysed
        :param matches: The matches to the previous keyframe
        :return: Nothing
        """
        # All tracks have not been updated
        for track in self.tracks:
            track.reset()

        new_tracks = []

        # For each match check if this feature already exists
        for point, correspondent in matches:
            # Convert to tuples
            point = (point[0], point[1])
            correspondent = (correspondent[0], correspondent[1])

            is_new_track = True

            for track in self.tracks:
                # If the current point matches the track's last point then they reference the same feature
                prior_point = track.getLastPoint()

                # So update the track
                if point == prior_point:
                    track.update(keyframe, correspondent)
                    is_new_track = False
                    break

            # Feature was not found elsewhere
            if is_new_track:
                new_tracks.append(Track(self.prev_keyframe_grey, point, keyframe, correspondent))

        # Add new tracks
        self.tracks += new_tracks

    def findRotationAndTranslation(self, matches):
        """
        Takes the matches between two frames and finds the rotation and translation of the second frame
        :param matches: The matched points between the frames
        :return: The rotation and translation matrix
        """
        # Convert matches into corresponding point vectors
        frame_left_points = np.ascontiguousarray(matches[:, 0])
        frame_right_points = np.ascontiguousarray(matches[:, 1])

        # Find essential matrix and inliers
        essential, mask = cv2.findEssentialMat(frame_left_points,
                                               frame_right_points,
                                               self.intrinsic,
                                               self.distortion,
                                               self.intrinsic,
                                               self.distortion)

        # TODO: Add distanceThresh for potential triangulated points
        # Use the essential matrix and inliers to find the pose
        _, R, t, _ = cv2.recoverPose(essential,
                                     frame_left_points,
                                     frame_right_points,
                                     self.intrinsic,
                                     mask=mask)

        return R, t
