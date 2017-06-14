# coding=utf-8
#from old_fslib.test import


import numpy as np
from ..old_fslib import EnvBuilder
from .loggers import SecondaryLogger
from .loggers import FastMotorActionsLogger
from .loggers import MarkovEnvLogger
from fslib.learning.loggers import TrialActionInfo



import os
import json
import sys

def ensure_dir(filename):
    """Check if directories in filename exists if not creates corresponding directories!"""
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)


def save_json_data(filename, data, rewrite=True):
    """Saves data in json format"""
    mode = "w" if rewrite else "a"

    ensure_dir(filename)
    with open(filename, mode) as the_file:
        json.dump(data, the_file)


class ValueFunction:
    """
    Класс адаптер к типу dict.
    """
    def __init__(self):
        self.func = {}

    def get(self, key):
        return self.func.get(key, None)

    def put(self, key, value):
        self.func[key] = value

    def add_if_absent(self, keys, value):
        for key in keys:
            if key not in self.func:
                self.func[key] = value

    def has(self, key):
        return key in self.func

    def clear(self):
        self.func.clear()


def is_fork(curr_sec, next_sec, net):
    """
    :param curr_sec: вторичная фс активированная во время испытания
    :param curr_sec: вторичная фс активированная в испытании непосредственно после активации curr_sec
    :param net: сеть функциональных систем
    :return:
    True - если активация curr_sec(вторичная фс) привела агента в состояние, в котором у него было выученно несколько
    стратегий поведения(из которых он выбрал next_sec)
    False - если, за исключением next_sec, не было других доступных выученных действий.
    """
    #assert isinstance(net, ln.BaseFSNetwork)
    #assert isinstance(next_sec, ln.BaseSecondary)
    cnet = net.get_cnet(next_sec.get_cnet_name())
    if cnet == None: return False
    variants = filter(lambda s: s != next_sec and s.AR_point()!= curr_sec.IA_point(), cnet.vertices)
    return len(variants) > 0


def update_action_vf_using_trial_log(logger, vf, net, alpha, reward=1.0, penalty=0.0):
    """
    Обновляет функцию ценности(vf) на основе данных(logger) о поведении агента(net) в испытании
    """
    r = range(0, len(logger.trial_history))  # длина trial_history равна числу действий предпринятых агентом в испытании
    r.reverse()
    th = logger.trial_history # trial_history - это лист структур TrialActionInfo
    next_step = None  # next - means that this step occurred later in the trial

    # is_good = True, если активация рассматриваемой вторичной фс в итоге привела агента к цели
    # is_good = False, если активация фс вела нас исключительно в направлении тупика
    is_good = True
    ends_list = [len(th) - 1]   # вспомогательные структуры.
    starts_list = []            #
    expected_reward = reward  # reward - финальное вознаграждение за достижение агентом цели.

    for i in r:  # рассматриваем последовательность действий агента начиная с последнего
        curr_step = th[i]
        #assert isinstance(curr_step, TrialActionInfo)
        # если цепочка действий под контролем вторичных систем закончилась останавливаем процесс.
        # По сути, мы обновлем функцию ценности только для последней в испытании непрерывной цепочки управляемых действий
        if curr_step.sec is None:
            break

        if next_step is not None: # если действие не было последним в испытании
            if curr_step.new_state is not None:  # если действие было успешным(переход в среде был доступен)
                if is_fork(curr_step.sec, next_step.sec, net):  # если curr_step.sec привело агента в состояние с выбором
                    expected_reward = vf.get(next_step.sec)

        if not is_good: # Если мы считаем что данная вторичная фс вела нас в тупик
            if starts_list[-1] > ends_list[-1]:
                starts_list.pop()
                ends_list.pop()
            try:
                j = starts_list[-1]
                if curr_step.sec.IA_point() == th[j].sec.AR_point():
                    starts_list[-1] += 1
                elif curr_step.new_state is not None:   # заходим в это условие как только вернулись из тупиковой ветви
                    is_good = True                      # пути к развилке из которой, впоследствии, вышли верным путем,
                    ends_list.append(i)                 # следоватльно фс ведушие к этой развилке вели нас к цели.
                    expected_reward = vf.get(th[j].sec)
            except IndexError:
                log = []
                for t in th:
                    mid = t.motor.get_id()
                    sid = t.new_state.get_id() if t.new_state is not None else None
                    sec = (t.sec.IA_point().get_id(), t.sec.AR_point().get_id()) if t.sec is not None else (None, None)
                    log.append((mid, sid, sec))
                save_json_data("error/trial_history.json", log)
                print("current index is {0}".format(i))
                save_json_data("error/trial_ends.json", ends_list)
                save_json_data("error/trial_starts.json", starts_list)
                raise


        if is_good and curr_step.new_state is None: # если мы угодили в тупик(не смогли пройти по ребру)
            is_good = False
            starts_list.append(i + 1)
            expected_reward = penalty  # штраф за то что уперлись в тупик

        val = vf.get(curr_step.sec)  # обновляем функцию ценности
        val += alpha * (expected_reward - val)  # коэффициент перед expected_reward считается равным 1
        vf.put(curr_step.sec, val)              # тк реальное вознаграждение в испытании присутствует только в конце.

        next_step = curr_step
    return vf


def update_motiv_to_secondary_weights(net, av, min_weight=0.98):
    """
    Функция устанавливает веса ребер от мотивационных фс к вторичным в соответствии со значениями функции ценности

    :param net: сеть функциональных систем
    :param av: функция ценности определенная на вторичных фс
    :param min_weight: минимальный вес ребра от мотивационной фс к вторичной
    """
    #assert isinstance(net, ln.BaseFSNetwork)

    for name in net.get_cnet_names():
        if name != net.MOTOR_NET and name != net.MOTIV_NET:
            cnet = net.get_cnet(name)
            values = map(lambda fs: av.get(fs), cnet.vertices)
            minn = np.min(values)
            maxx = np.max(values)
            for fs in cnet.vertices:
                denominator = maxx - minn
                val = 0.0
                if denominator != 0.0:
                    val = 0.01 * (av.get(fs) - minn)/denominator

                edge = next((e for e in fs.get_incoming() if e.get_src() == fs._motiv_fs), None)
                edge.set_weight(min_weight + val)


def mark_good_and_bad_secondary(logger):
    """
    Использовалась для тестированания механизма распространения награды за достижения цели
    и штрафов за поподание в тупик при пересчете функции ценности
    """
    r = range(0, len(logger.trial_history))
    r.reverse()
    th = logger.trial_history
    next_step = None

    vf = ValueFunction()
    is_good = True
    ends_list = [len(th) - 1]
    starts_list = []
    for i in r:
        curr_step = th[i]
        assert isinstance(curr_step, TrialActionInfo)
        if curr_step.sec is None:
            break

        if not is_good:
            if starts_list[-1] > ends_list[-1]:
                starts_list.pop()
                ends_list.pop()

            j = starts_list[-1]
            if curr_step.sec.IA_point() == th[j].sec.AR_point():
                starts_list[-1] += 1
            elif curr_step.new_state is not None:
                is_good = True
                ends_list.append(i)

        if is_good and curr_step.new_state is None:
            is_good = False
            starts_list.append(i + 1)

        if is_good: vf.put(curr_step.sec, 1)
        else: vf.put(curr_step.sec, -1)

    return vf


def mark_error_decay_secondary(logger, net):
    """
    Аналог сalc_action_value_function использовалась для тестирования алгоритма распространения ошибки
    """
    r = range(0, len(logger.trial_history))
    r.reverse()
    th = logger.trial_history
    next_step = None  # next - means that this step occurred later in the trial

    vf = ValueFunction()
    is_good = True
    ends_list = [len(th) - 1]
    starts_list = []
    expected_reward = 0

    for i in r:
        isfork = False
        curr_step = th[i]
        assert isinstance(curr_step, TrialActionInfo)
        if curr_step.sec is None:
            break

        if next_step is not None:
            if curr_step.new_state is not None:
                if is_fork(curr_step.sec, next_step.sec, net):
                    expected_reward =  vf.get(next_step.sec)

        if not is_good:
            if starts_list[-1] > ends_list[-1]:
                starts_list.pop()
                ends_list.pop()

            j = starts_list[-1]
            if curr_step.sec.IA_point() == th[j].sec.AR_point():
                starts_list[-1] += 1
            elif curr_step.new_state is not None:
                is_good = True
                ends_list.append(i)
                expected_reward = vf.get(th[j].sec)  #!!!

        if is_good and curr_step.new_state is None:
            is_good = False
            starts_list.append(i + 1)
            expected_reward = 0

        if is_good:
            vf.put(curr_step.sec, expected_reward + 1)
        else:
            vf.put(curr_step.sec, expected_reward - 1)

        next_step = curr_step
    return vf


class ChoiceStatistic(object):
    """
    Вспомогательный класс использовался при тестировании в модульных графах для того,
    чтобы собрать статистику по выбору агентом определенных вариантов.
    При тестировании на случайных графах не нужен. //перенести в /util

    """
    def __init__(self, states, states_names):
        self.list = []
        self.IA_AR_pairs = states
        self.variant_names = states_names

    def clear(self):
        self.list = []

    def reset(self, IA_AR_pair, variant_names):
        self.clear()
        self.IA_AR_pairs = IA_AR_pair
        self.variant_names = variant_names

    def check_statistic(self, logger):
        for info in logger.trial_history:
            if info.sec == None: continue
            for i in xrange(0, len(self.IA_AR_pairs)):
                if info.sec.IA_point() ==self.IA_AR_pairs[i][0] and info.sec.AR_point() == self.IA_AR_pairs[i][1]:
                    self.list.append(i)
                    return

    def get_stat(self, length = None):
        result = []
        if length is None:
            length = len(self.list)
        if length <= 0:
            return [None]*len(self.IA_AR_pairs)

        for i in xrange(0, len(self.IA_AR_pairs)):
            result.append( len(filter(lambda x: x == i ,self.list[-length:])) / float(length) )
        return result

    def print_stat(self, length = None):
        r = self.get_stat(length)
        for i in xrange(0, len(self.variant_names)):
            print("go " + self.variant_names[i]  + " first: ",  r[i])


