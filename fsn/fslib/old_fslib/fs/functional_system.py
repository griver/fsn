from ..graph.vertex import Vertex


class FunctionalSystem(Vertex):
    def __init__(self, name="FS", active_threshold = 0.990):
        """
        :arguments:
        name: name of this FS. Default name is "FS"
        active_threshold: if  FS considered active if current activity surpass threshold. Default name is


        """

        Vertex.__init__(self, name)

        self._cnet_name = None
        self._delta_si = None
        self._delta_ri = None
        self._delta_ci = None
        self._env = None

        self._S = 0
        self._R = 0
        self._C = 0
        self._IA = 0
        self._AR = 0
        self._I = 0
        self.active_threshold = active_threshold

        self._deactivated = False

    def is_active(self):
        return self._S >= self.active_threshold

    def recalculate_params(self):
        # calculate new parameters of fast_fs
        raise NotImplementedError('Method recalculate_params() is pure virtual')

    def apply_new_params(self):
        # self-describing name
        raise NotImplementedError('Method apply_new_params() is pure virtual')

    def deactivation_method(self):
        pass

    def get_S(self):
        """
        activity of the functional system
        """
        return self._S

    def get_R(self):
        """
        mismatch between current and goal states of the functional system
        """
        return self._R

    def get_C(self):
        """
        used for the deactivation of the system if it is ineffective
        """
        return self._C

    def get_IA(self):
        """
        corresponds to the recognition of the problem state
        """
        return self._IA

    def get_AR(self):
        """
        This parameter indicates the accomplishment of the goal.
        """
        return self._AR

    def get_I(self):
        return self._I

    def state(self):
        """
        :Returns:
        string representation of main FS internal parameters

        """
        result = str(self.get_id()) + " (" + self.name + ")\n"
        result += "  S: " + str(self.get_S())
        result += "  R: " + str(self.get_R())
        result += "  C: " + str(self.get_C())
        result += "  IA: " + str(self.get_IA())
        result += "  AR: " + str(self.get_AR())
        return result

    def reset(self):
        self._S = 0.
        self._R = 0.
        self._C = 0.
        self._IA = 0.
        self._AR = 0.
        self._I = 0.
        self._deactivated = False

    def is_deactivated(self):
        """
        returns self._deactivated value
        """
        return self._deactivated
    # --- --------------aliases---------------------------------------------

    def get_activity(self):
        """
        alias for get_S()
        """
        return self.get_S()

    def get_mismatch(self):
        """
        alias for get_R()
        """
        return self.get_R()

    def get_deactivation_variable(self):
        """
        alias for get_C()
        """
        return self.get_C()

    def get_input_afferentation(self):
        """
        alias for get_IA()
        """
        return self.get_IA()

    def get_result_acceptor(self):
        """
        alias for get_AR()
        """
        return self.get_AR()

    # -----------------utility methods---------------------------------------------
    def _calc_rk4_rs(self, h, env_interaction, influence_sum):
        K_1R = h * self._delta_ri(self._S, self._R, env_interaction)
        K_1S = h * self._delta_si(self._S, self._R, influence_sum)

        K_2R = h * self._delta_ri(self._S + 0.5 * K_1S, self._R + 0.5 * K_1R, env_interaction)
        K_2S = h * self._delta_si(self._S + 0.5 * K_1S, self._R + 0.5 * K_1R, influence_sum)

        K_3R = h * self._delta_ri(self._S + 0.5 * K_2S, self._R + 0.5 * K_2R, env_interaction)
        K_3S = h * self._delta_si(self._S + 0.5 * K_2S, self._R + 0.5 * K_2R, influence_sum)

        K_4R = h * self._delta_ri(self._S + K_3S, self._R + K_3R, env_interaction)
        K_4S = h * self._delta_si(self._S + K_3S, self._R + K_3R, influence_sum)

        delta_R = (K_1R + (2 * K_2R) + (2 * K_3R) + K_4R)/6
        delta_S = (K_1S + (2 * K_2S) + (2 * K_3S) + K_4S)/6
        return delta_R, delta_S


    def _calc_influence(self, pred):
        incoming = self.get_incoming()
        sum = 0
        for edge in incoming:
            if pred(edge):
                sum += edge.weight() * (edge.get_src().get_S())
        return sum

    def get_cnet_name(self):
        return self._cnet_name

    def set_cnet_name(self, competitive_network_name):
        self._cnet_name = competitive_network_name


class BaseMotor(FunctionalSystem):
    _motiv_cn = None

    def is_motivated(self):
        return self._motiv_cn.get_active() != None

    def change_coords(self):
        raise NotImplementedError()

    def edge_index(self):
        raise NotImplementedError()


class BaseSecondary(FunctionalSystem):
    _prev_state = None
    _goal_state = None
    _motiv_fs = None

    def IA_point(self):
        return self._prev_state

    def AR_point(self):
        return self._goal_state

    def get_motivation(self):
        return self._motiv_fs


class BaseMotivational(FunctionalSystem):
    _goal_state = None

    def get_goal(self):
        return self._goal_state


