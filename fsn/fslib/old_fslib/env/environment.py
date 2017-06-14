# coding=utf-8
from .. import Graph
from ..graph import algo

class Environment(Graph):
    def __init__(self, dimension, name="environment", start_state_id=0,):
        Graph.__init__(self, name)
        self._dim = dimension # the dimension of a space
        self._curr_state = None
        self._start_id = start_state_id

    def get_dimension(self):
        return self._dim

    def distance(self, src, dst):
        #if(src.get_dimension() != dst.get_dimension()):
        dist = 0
        for i in xrange(0, src.get_dimension()):
            dist += abs(src.coords()[i] - dst.coords()[i])

        return dist**(1.0/2.0)  # Euclidean norm

    def reset(self):
        if len(self.vertices()) > self._start_id:
            self.set_current_state(self.get_vertex(self._start_id))

    def distance_from_current(self, state):  # get distance from current environment state
        return self.distance(self.get_current_state(), state)

    def get_state_by_coords(self, coords):
        if len(coords) != self.get_dimension():
            raise ValueError("invalide number of coordinates")
        return next((p for p in self._vertex_list if (p.coords() == coords)), None)

    def get_current_state(self):
        #if self._curr_state is None:
        #    self.reset()
        return self._curr_state

    def set_current_state(self, state):
        if self.is_contains_vertex(state):
            self._curr_state = state
            return True
        return False

    def has_path_to(self, target_state):
        return target_state is algo.bfs(self, self.get_current_state(), target_state)

    def update_state(self, index):
        curr = self.get_current_state()
        tmp_st = None

        #ищем вершину с заданными координатами среди смежных вершин
        #for e in curr.get_outcoming():
        #    if e.get_dst().coords() == coords:
        if index < len(curr.get_outcoming()):
            e = curr.get_outcoming()[index]
            if e.is_available():
                tmp_st = e.get_dst()

        if tmp_st is None:
            #if self._msg:
                #print("Нельзя перейти из " + curr.name + " в " + str(coords))
            #    self._msg = False
            return

        #print("Перешли из " + curr.name + " в " + tmp_st.name)
        self.set_current_state(tmp_st)
        #self._msg = True


