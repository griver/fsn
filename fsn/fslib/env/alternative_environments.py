# coding=utf-8
from ..old_fslib import StochasticEnvironment
from  ..old_fslib import Environment
import numpy as np

class MarkovEnvironment(StochasticEnvironment):
    """
    Стохастическая среда, где успшеность каждого действия зависит только от текущего состояния
    """

    def update_state(self, index):
        """
        Переводит агента в новое состояние если выбранное агентом действие доступно и после этого
        обновляет доступность переходов соответствующи текущему состоянию среды.
        """
        super(MarkovEnvironment, self).update_state(index)
        self._update_transits()

    def reset(self):
        """
        Переводит среду в начальное состояние и обновляет доступность переходов из него
        """
        Environment.reset(self)
        self._update_transits()

    def has_path_to(self, target_state):
        return True

    def _update_transits(self):
        edges = self.get_current_state().get_outcoming()
        for e in edges:
            e.get_stochastic_group().recalc()


class ChangeStateStochasticityEnv(StochasticEnvironment):
    """
    Стохасчтическая среда, где доступность ребер разыгрывается после каждого обновления состояния среды
    """

    def set_current_state(self, state):
        """
        Переводит среду в новое состояние.
        Если состояние изменилось, то пересчитывает для нового состояние среды доступность переходов связанных с ним.
        """
        previous = self.get_current_state()
        result = Environment.set_current_state(self, state)
        current = self.get_current_state()
        if current is not previous:  # На данный момент состояние меняется с каждым вызовом
            self._recalc_transits()  # функции, но преждевременная оптимизация корень всех бед.

        return result

    def has_path_to(self, target_state):
        # в данной среде путь доступен всегда за исключением случая когда мы оказались заперты в текущем состоянии
        return any(e.is_available() for e in self.get_current_state().get_outcoming())


    def reset(self):
        """
        Возвращает среду в начальное состояние и обязательно разыгрывает вероятности переходов из стартового состояния
        """
        Environment.reset(self)
        self._recalc_transits()

    def _recalc_transits(self):
        edges = self.get_current_state().get_outcoming()
        any_available = False

        while True:  # с таким способом пересчета агент может застрять запертым в состоянии
            for e in edges:
                e.get_stochastic_group().recalc() # возможно стоит определить recalc и в классе ребра
                any_available = True if e.is_available() else any_available  # True if any

            if any_available:  # если из состояния можно выбраться завершаем
                    break


class PeriodicEnvironment(StochasticEnvironment):
    """
    Стохастическая среда, где успешность действий разыгрывается с определенной частотой
    """

    def __init__(self, period, dimension, name="PeriodicEnvironment", start_state_id = 0):
        super(PeriodicEnvironment, self).__init__(dimension, name, start_state_id)
        self.period = max(1, period)
        self._counter = 0

    def update_state(self, index):

        super(PeriodicEnvironment, self).update_state(index)

        self._counter += 1
        if self._counter >= self.period:
            self._counter = 0
            self._update_all_transits()

    def _update_all_transits(self):
        for g in self._stoch_groups:
            g.recalc()

    def reset(self):
        """ Сейчас переводит среду в начальное состояние и обновляет доступность всех переходов """
        super(PeriodicEnvironment, self).reset()
        self._counter = 0

    def has_path_to(self, target_state):
        if self.period > np.sqrt(self.get_vertices_number()):
            return super(PeriodicEnvironment, self).has_path_to(target_state)
        return True  # среда обновляется с переодичностью так, что рано или поздно путь откоется.
