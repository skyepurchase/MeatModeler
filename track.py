class Track:
    def __init__(self, prev_frame_ID, feature, frame_ID, correspondent, point):
        """
        tracks a 3D point through different frames

        :param prev_frame_ID: The frame number of the previous frame
        :param feature: The coordinates of the captured 3D point
        :param frame_ID: The frame number of the current frame
        :param correspondent: The coordinates of the corresponding feature
        :param point: The corresponding 3D point
        """
        self.coordinates = {prev_frame_ID: feature,
                            frame_ID: correspondent}
        self.points = [point]

    def update(self, frame_ID, correspondent, point):
        self.coordinates[frame_ID] = correspondent
        self.points.append(point)

    def getCoordinate(self, frame_ID):
        return self.coordinates.get(frame_ID)

    def getCoordinates(self):
        return self.coordinates

    def getPoint(self):
        """
        :return: The original triangulated point
        """
        return self.points[0]

    def getFinalPoint(self):
        """
        :return: The most disparate frames
        """
        return self.points[-1]
