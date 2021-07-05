class Track:
    def __init__(self, prev_frame, point, frame, correspondent):
        self.first_frame = prev_frame
        self.last_frame = frame
        self.virtual_point_vector = [point, correspondent]
        self.physical_point = None
        self.updated = True

    def update(self, frame, correspondent):
        self.last_frame = frame
        self.virtual_point_vector.append(correspondent)
        self.updated = True

    def reset(self):
        self.updated = False

    def getPoints(self):
        return self.virtual_point_vector

    def getTriangulationData(self):
        return self.first_frame, self.last_frame, self.virtual_point_vector[0], self.virtual_point_vector[-1]

    def setPhysicalPoint(self, point):
        self.physical_point = point
