# coding=utf-8
from ..graph.graph import  Graph
from .functional_system import BaseMotor
from .functional_system import BaseSecondary
from .functional_system import BaseMotivational


class CompetitiveNetwork(object):
    def __init__(self, edge_weight):
        self.e_weight = edge_weight
        self.vertices = []

    def add_fs(self, fs):
        self.vertices.append(fs)

    def get_active(self):
        for v in self.vertices:
            if v.is_active():
                return v
        return None



class BaseFSNetwork(Graph):


    def __init__(self, motor_edges_weight, sec_edges_weight, motiv_edges_weight,  name="FSNetwork"):
        Graph.__init__(self, name)
        self.MOTOR_NET = "MOTOR"
        self.MOTIV_NET = "MOTIV"
        self.time = 0

        self._act_weight = motor_edges_weight
        self._motiv_weight = motiv_edges_weight
        self._sec_weight = sec_edges_weight
        self._curr_action = None

        self._cnets = {}
        self.create_cnet(self.MOTOR_NET, motor_edges_weight)
        self.create_cnet(self.MOTIV_NET, motiv_edges_weight)


        self.__get_action_last_call = 0
        self._last_action = None

    def reset(self):
        self.time = 0
        for fs in self._vertex_list:
            fs.reset()

    def recalc(self):
        for fs in self._vertex_list:
            fs.recalculate_params()

    def apply(self):
        self.time += 1
        for fs in self._vertex_list:
            fs.apply_new_params()

    def add_motor(self, fs):
        self.add_in_cnet(fs, self.MOTOR_NET)

    def add_motiv(self, fs):
        self.add_in_cnet(fs, self.MOTIV_NET)

    def all_motor(self):
        return self.filter_by_type(BaseMotor)

    def all_secondary(self):
        return self.filter_by_type(BaseSecondary)

    def all_motiv(self):
        return self.filter_by_type(BaseMotivational)


    def get_action(self):
        if self.__get_action_last_call == self.time:
            return self._curr_action

        result = []
        for fs in self.all_motor():
            if fs.is_active():
                result.append(fs)

        if len(result) > 1:
            raise ValueError("Одновременно активны несколько FS действия")
        elif len(result) == 0:
            self._curr_action = None
        else:
            self._curr_action = result[0]
            self._last_action = self._curr_action

        self.__get_action_last_call = self.time

        return self._curr_action

    def get_last_action(self):
        return self._last_action

    def filter_by_type(self, type):
        return filter(lambda v: isinstance(v, type), self.vertices())

    def find_fs(self, predicate):
        return filter(predicate, self.vertices())

    def create_cnet(self, name, edge_weight):
        if name in self._cnets:
            raise ValueError("Competitive network with this name already exist")
        self._cnets[name] = CompetitiveNetwork(edge_weight)

    def get_cnet(self, name):
        return self._cnets.get(name, None)

    def get_cnet_names(self):
        return self._cnets.keys()

    def add_in_cnet(self, fs, cnet_name):
        assert fs.get_cnet_name() is None
        id = self.add_vertex(fs)
        cnet = self.get_cnet(cnet_name)

        if fs in cnet.vertices:
            raise ValueError("Competitive network already include this functional system")

        for v in cnet.vertices:
            self.add_edge(fs, v, cnet.e_weight)
            self.add_edge(v, fs, cnet.e_weight)

        cnet.add_fs(fs)
        fs.set_cnet_name(cnet_name)
        return id