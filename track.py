class Track:
    def __init__(self, prev_frame, point, frame, correspondent):
        self.first_frame_ID = prev_frame
        self.last_frame_ID = frame
        self.point_colour = None
        self.virtual_point_vector = [point, correspondent]
        self.physical_point = None

    def update(self, frame, correspondent):
        self.last_frame_ID = frame
        self.virtual_point_vector.append(correspondent)

    def getTriangulationData(self):
        return self.first_frame_ID, self.last_frame_ID, self.virtual_point_vector[0], self.virtual_point_vector[-1]

    def getColor(self):
        return self.point_colour

    def setPhysicalPoint(self, point):
        self.physical_point = point
