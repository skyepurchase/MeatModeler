class Track:
    def __init__(self, prev_frame_ID, point, frame_ID, correspondent, physical_point):
        self.first_frame_ID = prev_frame_ID
        self.last_frame_ID = frame_ID
        self.virtual_point_vector = [point, correspondent]
        self.physical_point_vector = [physical_point]
        self.updated = False

    def update(self, frame_ID, correspondent, point):
        self.last_frame_ID = frame_ID
        self.virtual_point_vector.append(correspondent)
        self.physical_point_vector.append(point)
        self.updated = True

    def reset(self):
        self.updated = False

    def getLastPoint(self):
        return self.virtual_point_vector[-1]

    def getTriangulationData(self):
        return self.first_frame_ID, self.last_frame_ID, self.virtual_point_vector, self.physical_point_vector

    def wasUpdated(self):
        return self.updated
