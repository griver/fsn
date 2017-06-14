
class Vertex(object):
    def __init__(self, name="none"):
        self.__id = -1
        self._incoming = []
        self._outcoming = []
        self.name = name


    def get_id(self):
        return self.__id

    def get_outcoming(self):
        return tuple(self._outcoming)

    def get_incoming(self):
        return tuple(self._incoming)

    def _set_id(self, id):  # only for graph class internal usage
        self.__id = id

    def _add_outcoming(self, edge):  # only for graph class internal usage
        if self._outcoming.count(edge) == 0:
            self._outcoming.append(edge)

    def _add_incoming(self, edge):  # only for graph class internal usage
        if self._incoming.count(edge) == 0:
            self._incoming.append(edge)

    def _remove_outcoming(self, edge):
        self._outcoming.remove(edge)

    def _remove_incoming(self, edge):
        self._incoming.remove(edge)

    def __str__(self):
        return str(self.get_id())
