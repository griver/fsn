from ..functional_system import BaseMotivational

class LMMotivation(BaseMotivational):
    def __init__(self, env, goal_state):
        BaseMotivational.__init__(self, "LMMotiv(" + goal_state.name + ')')
        self._env = env
        self._goal = goal_state

    def recalculate_params(self):
        pass

    def apply_new_params(self):
        pass
