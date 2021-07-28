class Track:
    def __init__(self, prev_frame_ID, point, frame_ID, correspondent):
        self.first_frame_ID = prev_frame_ID
        self.last_frame_ID = frame_ID
        self.virtual_point_vector = [point, correspondent]

    def update(self, frame_ID, correspondent):
        self.last_frame_ID = frame_ID
        self.virtual_point_vector.append(correspondent)

    def getTriangulationData(self):
        return self.first_frame_ID, \
               self.last_frame_ID, \
               self.virtual_point_vector[0], \
               self.virtual_point_vector[-1]

    def get2DPoints(self):
        return self.virtual_point_vector
