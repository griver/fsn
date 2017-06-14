import random

from ..functional_system import *


class SecondaryFS(BaseSecondary):

    def __init__(self, env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_fs, prev_state, goal_state):

        assert isinstance(motiv_fs, BaseMotivational)
        BaseSecondary.__init__(self, "Sec_"+ str(motiv_fs.get_id()) + ":" + prev_state.name + "->" + goal_state.name)

        self._newS = 0
        self._newR = 0
        self._newC = 0
        self._newIA = 0
        self._newAR = 0
        self._newI = 0

        self._env = env
        self._motiv_fs = motiv_fs
        self._prev_state = prev_state
        self._goal_state = goal_state

        self._delta_si = delta_si
        self._delta_ri = delta_ri
        self._delta_ci = delta_ci
        self._calc_IA = calc_IA
        self._calc_AR = calc_AR
        self._calc_ii = calc_ii

        self._active_time = 0

    def recalculate_params(self):  # calculate new parameters of fast_fs

        ia_dist = self._env.distance_from_current(self._prev_state)
        ar_dist = self._env.distance_from_current(self._goal_state)

        self._newIA = self._calc_IA(self._motiv_fs.get_S(), ia_dist)
        self._newAR = self._calc_AR(ar_dist)

        influence_sum = self._calc_influence(lambda x: True)
        self._newI = self._calc_ii(self)

        delta_R, delta_S = self._calc_rk4_rs(1, self._newI, influence_sum)
        self._newR = self._R + delta_R + 0.001 * random.random() * (self._newIA > 0.02)
        self._newS = self._S + delta_S + 0.001 * random.random() * (self._newIA > 0.02)

        self.deactivation_method()

    def apply_new_params(self):  # self-describing name
        self._S = self._newS
        self._R = self._newR
        self._C = self._newC
        self._IA = self._newIA
        self._AR = self._newAR


    def deactivation_method(self):
        if self.is_active() and not self._deactivated:
            #print("Set deactivation to TRUE (" + self.name + ")")
            self._deactivate = True
            self.__dnumber = 0

        if self._deactivated:
            self.__dnumber += 1
            #print(self.name + ": dnumber = " + str(self.__dnumber))
            if self.__dnumber == 10:
                #print(self.name + " is deactivated!")
                self._newR = 0.0
                self._newS = 0.0
