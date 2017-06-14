# coding=utf-8
class Edge(object):
    def __init__(self, src, dst, weight):  # возможно стоит поменять weight на data
        self._src = src
        self._dst = dst
        self._weight = weight

        src._add_outcoming(self)
        dst._add_incoming(self)

    def get_src(self):
        return self._src

    def get_dst(self):
        return self._dst

    def weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def remove(self):
        self._src._remove_outcoming(self)
        self._dst._remove_incoming(self)

    def __str__(self):
        result = "Src:" + str(self._src.get_id()) + " Dst:" + str(self._dst.get_id()) + " Weight:" + str(self._weight)

    def is_available(self):
        return True