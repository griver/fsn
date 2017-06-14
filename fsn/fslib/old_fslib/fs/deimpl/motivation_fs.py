from ..functional_system import *
from ...env.point import Point
import random

class SimpleMotivFS(BaseMotivational):
    def __init__(self, env, goal_state):
        BaseMotivational.__init__(self, "SimpleMotiv(" + goal_state.name + ')')
        self._goal_state = goal_state
        self._S = 1
        self._newS = 1
        self._env = env


    def recalculate_params(self): # calculate new parameters of fast_fs
        if isinstance(self._env.get_current_state(), Point):
            if self._goal_state == self._env.get_current_state():
                self._newS = 0
            else:
                self._newS = 1

    def apply_new_params(self):  # self-describing name
        self._S = self._newS

    def reset(self):
        self._S = 1
        self._newS = 1

    def set_goal(self, point):
        self._goal_state = point


class MotivationFS(BaseMotivational):

    def __init__(self, env, integration_function, delta_si, delta_ri, calc_IA, calc_AR, calc_I, goal_state):
        BaseMotivational.__init__(self, "Motiv(" + str(goal_state.get_id()) + ')')
        self._goal_state = goal_state
        self._env = env

        self._integration = integration_function
        self._delta_si = delta_si
        self._delta_ri = delta_ri
        self._calc_IA = calc_IA
        self._calc_AR = calc_AR
        self._calc_ii = calc_I

    def recalculate_params(self):
        self._newIA = self._calc_IA(self)
        self._newAR = self._calc_AR(self)
        self._newI = self._calc_ii(self)

        influence_sum = self._calc_influence(lambda x: True)

        delta_R2, delta_S2 = self._integration(self)
        #self._newR = self._R  + delta_R
        #self._newS = self._S  + delta_S
        delta_R, delta_S = self._calc_rk4_rs(1, self._newI, influence_sum)
        if isinstance(self, BaseMotivational):
            i = 55

        self._newR = self._R + delta_R + 0.001 * random.random() * (self._newIA > 0.02)
        self._newS = self._S + delta_S + 0.001 * random.random() * (self._newIA > 0.02)

        self.deactivation_method()

    def apply_new_params(self):
        self._S = self._newS
        self._R = self._newR
        self._IA = self._newIA
        self._AR = self._newAR
        self._I = self._newI
