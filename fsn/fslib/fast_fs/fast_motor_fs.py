# coding=utf-8
from ..old_fslib import BaseMotor, BaseFSNetwork, Environment

import random as rnd


def build_fast_motor(env, net, index):
    """
    Build FastMotor FS. If the FS is active then agent must go through outcoming edge with corresponding index.
    """
    return FastMotor(env, net, net.get_cnet(net.MOTIV_NET), index)

class FastMotor(BaseMotor):
    """
    Пропускает честный расчет активности FS. И делает случайный равновероятный выбор между одной из имеющихся моторных
    ФС, если для данного состояния нет вторичных функциональных ситем, которые могли бы управлять поведением агента.
    """
    _active_index = None
    _params_applied = True
    _is_controlled = False

    @staticmethod
    def recalc_params(fast_fs):
        if not FastMotor._params_applied: return # перерасчет выполняется только один раз между вызовами apply_params
        FastMotor._params_applied = False
        FastMotor._active_index = None  # Подумать еще об этом.

        #assert isinstance(fast_fs, FastMotor)
        net = fast_fs._net
        env = fast_fs._env
        #assert isinstance(env, Environment)
        #assert isinstance(net, BaseFSNetwork)

        secs = [sec for sec in net.all_secondary()
                if sec.IA_point() is env.get_current_state() and sec.is_deactivated() is False]

        if len(secs) > 0:  # Если выученны вторичные фс связанные с этим состоянием, то ждем пока одна из них не победит
            active_sec = next((s for s in secs if s.is_active()), None)
            FastMotor._is_controlled = True

            if active_sec is not None:
                edges = env.get_current_state().get_outcoming()
                for i in xrange(len(edges)): # else для отлавливания ошибок если Break не случился. Случаться он должен всегда.
                    if edges[i].get_dst() == active_sec.AR_point():
                        FastMotor._active_index = i
                        break

        else:  # Если нет  вторичных ФС случайно выбираем действие
            FastMotor._active_index = rnd.randrange(len(env.get_current_state().get_outcoming()))
            FastMotor._is_controlled = False

    @staticmethod
    def apply_params(fast_fs): # на самом деле, нет смысла делать статическим.
        FastMotor._params_applied = True
        #assert isinstance(fast_fs, FastMotor)
        # здесь мы учитываем что переход осуществляется указанием индекса ребра перехода,
        # а не изменением вектора состояния.

        # active if fast_fs._S >= fast_fs.active_threshold
        fast_fs._S = fast_fs.active_threshold*(fast_fs.edge_index() == FastMotor._active_index)*fast_fs.is_motivated()

    @staticmethod
    def is_controlled():
        """
        :return: True - if Motor FS network is controlled by any secondary FS. False - otherwise.
        """
        return FastMotor._is_controlled

    def __init__(self, env, net, motiv_cn, index):
        BaseMotor.__init__(self, "Motor{0}".format(index))
        self._newS = 0
        self._newR = 0
        self._newC = 0
        self._newIA = 0
        self._newAR = 0
        self._newI = 0

        self._env = env
        self._net = net
        self._motiv_cn = motiv_cn
        self._index = index

    def recalculate_params(self):
        FastMotor.recalc_params(self)

    def apply_new_params(self):
        FastMotor.apply_params(self)

    def edge_index(self):
        return self._index