from .. import Vertex

class Point(Vertex):
    def __init__(self, coordinates):
        Vertex.__init__(self, coordinates.__str__())
        self._coord = coordinates
        self.name = str(coordinates)

    def coords(self):
        return self._coord

    def get_dimension(self):
        return len(self._coord)

    def __repr__(self):
        return self.name

