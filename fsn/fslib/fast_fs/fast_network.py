# coding=utf-8
from ..old_fslib import BaseFSNetwork, FSBuilder


def create_fast_network(env, create_motor, name = "FastFSNetwork"):
    """
    Create fast FS network with edge moving motor systems
    """
    net = FastFSNetwork(FSBuilder.act_edge, FSBuilder.sec_edge, FSBuilder.motiv_edge, name)

    max_degree = max(len(v.get_outcoming()) for v in env.vertices())

    for i in xrange(max_degree):
        mot0 = create_motor(env, net, i)
        net.add_motor(mot0)

    net.set_env(env)
    return net

class FastFSNetwork(BaseFSNetwork):
    """
    Ускоряет пересчет активности слоя вторичных функциональных систем,
    расчитывая значения только тех фс активность которых действительно может измениться на текущем шаге.
    """
    def __init__(self, motor_edges_weight, sec_edges_weight, motiv_edges_weight, name="FSNetwork"):
        super(FastFSNetwork, self).__init__(motor_edges_weight, sec_edges_weight, motiv_edges_weight, name)
        self._all_motor = []
        self._all_motiv = []
        self._all_sec = []
        self._env = None
        self.max_sec_in_step = 0

    def recalc(self):
        if not self._check_lists_consistency():
            self._reset_fs_lists()

        if not self._check_lists_consistency():
            fast_number =  len(self._all_motor) + len(self._all_sec) + len(self._all_motiv)
            real_number = len(self._vertex_list)
            raise Exception("Fast_network:: vertex count failure: {0} != {1}".format(fast_number, real_number))

        for m in self._all_motiv:
            m.recalculate_params()

        for m in self._all_motor:
            m.recalculate_params()

        i = 0
        state = self._env.get_current_state()
        for s in self._all_sec:
            if state is not None: # на всякий случай для работы с предсказнием. где у нас не будет среды
                if state is s.IA_point() or state is s.AR_point() or s.get_S() > 0.01:
                    s.recalculate_params()
                    i += 1
            else:
                s.recalculate_params()

        self.max_sec_in_step = max(i, self.max_sec_in_step)

    def reset(self):
        super(FastFSNetwork, self).reset()
        self._reset_fs_lists()

    def _reset_fs_lists(self):
        self._all_sec = super(FastFSNetwork,self).all_secondary()
        self._all_motor = super(FastFSNetwork,self).all_motor()
        self._all_motiv = super(FastFSNetwork,self).all_motiv()

    def set_env(self, env):
        self._env = env

    def all_secondary(self):
        if not self._check_lists_consistency():
            self._reset_fs_lists()
        return self._all_sec

    def all_motiv(self):
        if not self._check_lists_consistency():
            self._reset_fs_lists()
        return self._all_motiv

    def all_motor(self):
        if not self._check_lists_consistency():
            self._reset_fs_lists()
        return self._all_motor

    def _check_lists_consistency(self):
        return len(self._all_motor) + len(self._all_sec) + len(self._all_motiv) == len(self._vertex_list)