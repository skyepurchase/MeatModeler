class Track:
    def __init__(self, prev_frame_ID, feature, frame_ID, correspondent, point=None):
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
        self.point = point
        self.updated = True

    def reset(self):
        self.updated = False

    def wasUpdated(self):
        return self.updated

    def getTriangulationData(self):
        keys = list(self.coordinates.keys())
        return keys[0], keys[-1], self.coordinates.get(keys[0]), self.coordinates.get(keys[-1])

    def get2DPoints(self):
        return list(self.coordinates.values())

    def update(self, frame_ID, correspondent):
        self.coordinates[frame_ID] = correspondent
        self.updated = True

    def getCoordinate(self, frame_ID):
        return self.coordinates.get(frame_ID)

    def getCoordinates(self):
        return self.coordinates

    def setPoint(self, point):
        self.point = point

    def getPoint(self):
        return self.point
