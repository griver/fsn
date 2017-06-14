# coding=utf-8
from ..old_fslib import TrialLogger
from .. import old_fslib

class TrialActionInfo:
    """
    TrialActionInfo - структура для хранения информации о действии агента во время испытания.
    :motor - действие(соответствующая моторная фс)
    :new_state - новое состояние в которое перешел агент, если действие не удалось то None
    :sec - вторичная фс которая побудила агент к действию, если таковой не было то None
    """
    def __init__(self, motor = None, new_state = None, secondary = None):
        self.motor = motor
        self.sec = secondary
        self.new_state = new_state

    def __repr__(self):
        str = "mot: " +  self.motor.name
        if self.sec != None:
            str += "|sec: " + self.sec.name
        if self.new_state != None:
            str += "|st: " + repr(self.new_state)
        return str



class CompetitiveNetworkError(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)


class SecondaryLogger(TrialLogger):
    """
    Расширение TrialLogger. Для каждого действия агента(активация моторной фс) заполняет структуру TrialActionInfo.
    После испытания эта информация используется для обновления функции ценности
    """
    def __init__(self):
        TrialLogger.__init__(self)
        self.trial_history = []
        self._prev_action = None
        self._prev_state = None

    def add(self, net, env):
        TrialLogger.add(self,net, env)

        #assert isinstance(net, old_fslib.BaseFSNetwork)
        action = net.get_action()
        if action != self._prev_action:
            self._prev_action = action
            if action is not None:
                sec = self._get_secondary(net, action)
                if self._prev_state == env.get_current_state():
                    info = TrialActionInfo(action, None, sec)
                else:
                    info = TrialActionInfo(action, env.get_current_state(), sec)
                    self._prev_state = env.get_current_state()
                self.trial_history.append(info)

    def start_trial(self, env, net):
        TrialLogger.start_trial(self, env, net)
        self.trial_history = []
        self._prev_action = None
        self._prev_state = env.get_current_state()

    #def reset(self):
    #    self.clear()

    def clear(self):
        TrialLogger.clear(self)
        self.trial_history = []
        self._prev_action = None
        self._prev_state = None

    @staticmethod
    def _get_secondary(net, motor):
        assert isinstance(motor, old_fslib.BaseMotor)
        secs = filter(lambda e: isinstance(e.get_src(), old_fslib.BaseSecondary) and e.get_src().is_active(), motor.get_incoming())
        if len(secs) == 0:
            return None
        elif len(secs) > 1:
            print "активные вторичные фс:"
            for s in secs:
                print s.get_src().IA_point(), " -> ", s.get_src().AR_point(), s.get_src().get_activity()
            raise CompetitiveNetworkError("Несколько вторичных фс, вероятно из одной конурентой сети, одновременно активны!")
        else:
            return secs[0].get_src()

    def get_actions_number(self):
        return len(self.trial_history)


class FastMotorActionsLogger(SecondaryLogger):
    def add(self, net, env):
        TrialLogger.add(self,net, env)

        assert isinstance(net, old_fslib.BaseFSNetwork)
        action = net.get_action()

        if action is not None:
            #Если действие контролировалось вторичными фс, то оно может длиться некторое время
            # и это действие следует считать тем же самым
            repeated_action = action.is_controlled() and action == self._prev_action
            if not repeated_action:
                sec = self._get_secondary(net, action)
                if self._prev_state == env.get_current_state():
                    info = TrialActionInfo(action, None, sec)
                else:
                    info = TrialActionInfo(action, env.get_current_state(), sec)
                    self._prev_state = env.get_current_state()
                self.trial_history.append(info)

        # каждый раз запоминаем какое действие было до этого
        self._prev_action = action


class MarkovEnvLogger(SecondaryLogger):
    def add(self, net, env):
        """
        В случае марковского процесса принятия решений каждый момент когда моторная система активна
        она может считаться отдельным действием.
        """
        TrialLogger.add(self,net, env)
        assert isinstance(net, old_fslib.BaseFSNetwork)
        action = net.get_action()

        if action is not None:
            sec = self._get_secondary(net, action)
            if self._prev_state == env.get_current_state():
                info = TrialActionInfo(action, None, sec)
            else:
                info = TrialActionInfo(action, env.get_current_state(), sec)
                self._prev_state = env.get_current_state()
            self.trial_history.append(info)

        self._prev_action = action