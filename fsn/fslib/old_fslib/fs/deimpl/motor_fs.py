
import random

from ..functional_system import *

# MotorFS trying to change env.current_state[index] to value dst.

class MotorFS(BaseMotor):
    def __init__(self, env, delta_si, delta_ri, delta_ci, calc_IA, calc_AR, calc_ii, motiv_cn, index, goal):
        BaseMotor.__init__(self, "Motor"+ str(index) + str(goal))
        self._newS = 0
        self._newR = 0
        self._newC = 0
        self._newIA = 0
        self._newAR = 0
        self._newI = 0

        self._env = env
        self._motiv_cn = motiv_cn
        self._index = index
        self._goal = goal

        self._delta_si = delta_si
        self._delta_ri = delta_ri
        self._delta_ci = delta_ci
        self._calc_IA = calc_IA
        self._calc_AR = calc_AR
        self._calc_ii = calc_ii
        self._active_time = 0
        self.__dnumber = 0


    def recalculate_params(self):
        self._newIA = self._calc_IA(self)
        self._newAR = self._calc_AR(self)
        self._newI = self._calc_ii(self)

        influence_sum = self._calc_influence(lambda e: isinstance(e.get_src(), BaseMotor))
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
        self._I = self._newI


    def deactivation_method(self):
        if self.is_active() and not self._deactivated:
            self._deactivated = True
            #print("Set deactivation to TRUE (" + self.name + ")")
            self.__dnumber = 0  # """

        #if self._deactivated == True:
            #print(self.name + ": __dnumber = " + str(self.__dnumber))

        if self._deactivated:
            self.__dnumber += 1
            if self.__dnumber == 10:
                #print("Set deactivation to FALSE (" + self.name + ")")
                self._deactivated = False
                self._newR = 0.0
                self._newS = 0.0

    def change_coords(self):
        coords = self._env.get_current_state().coords()
        c = list(coords)
        c[self._index] = self._goal
        return tuple(c)

    def edge_index(self):
        return self._index

class AdditiveMotorFS(MotorFS):
    def change_coords(self):
        coords = self._env.get_current_state().coords()
        c = list(coords)
        change = self._goal*2 - 1
        c[self._index] += change
        return tuple(c)