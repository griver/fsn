# coding=utf-8
from ..old_fslib import Environment
import numpy as np
from . import stochastic_learning as sln


def epsilon_greedy_strategy(epsilon):
    """
    Возвращает функцию которая, принимая состояние среды и функцию ценности, делает epsilon-жадный выбор действия.
    """
    def epsilon_greedy_choice(vf, state):
        actions = state.get_outcoming()
        is_greedy = np.random.rand() >= epsilon

        if is_greedy:
            i = np.argmax(map(lambda x: vf.get(x), actions))
        else:
            i = np.random.randint(0, len(actions))

        return actions[i]

    return epsilon_greedy_choice


greedy_strategy = epsilon_greedy_strategy(0.0)

MAX_ACTIONS_NUMBER = 15000

def SARSA(env, vf, goal, strategy, alpha, gamma, reward, penalty):
    assert isinstance(env, Environment)
    assert isinstance(vf, sln.ValueFunction)

    actions_number = 0
    action = strategy(vf, env.get_current_state())
    new_action = None

    while env.get_current_state() is not goal:
        if actions_number >= MAX_ACTIONS_NUMBER: # for testing torus environment
            break

        index = env.get_current_state().get_outcoming().index(action)
        action_success = action.is_available()  # Она находиться здесь тк доступности разыгрываются для нового состояния
                                                # После действия либо не рызгрываются вообще, поэтому мы знвем успех
                                                # действия заранее. Если это измениться вдруг, нужно будет менять и здесь
        env.update_state(index)
        actions_number += 1
        next_vf = 0.0
        rw = 0.0

        if env.get_current_state() is goal:
            rw = reward
        else:
            if not action_success:
                rw = penalty
            new_action = strategy(vf, env.get_current_state())
            next_vf = vf.get(new_action)

        val = vf.get(action)
        val += alpha * (rw + gamma * next_vf - val)

        vf.put(action, val)
        action = new_action

    #print("number of actions = " + str(actions_number))
    return actions_number


def Q_learning(env, vf, goal, strategy, alpha, gamma, reward, penalty):
    assert isinstance(env, Environment)
    assert isinstance(vf, sln.ValueFunction)

    actions_number = 0

    while env.get_current_state() is not goal:
        if actions_number >= MAX_ACTIONS_NUMBER:  # for testing torus environment
            break

        action = strategy(vf, env.get_current_state())
        index = env.get_current_state().get_outcoming().index(action)
        action_success = action.is_available()  # Она находиться здесь тк доступности разыгрываются для нового состояния
                                                # После действия либо не рызгрываются вообще, поэтому мы знвем успех
                                                # действия заранее. Если это измениться, вдруг нужно будет менять и здесь
        env.update_state(index)
        actions_number += 1
        next_vf = 0.0
        rw = 0.0

        if env.get_current_state() is goal:
            rw = reward
        else:
            if not action_success:
                rw = penalty
            best_action = greedy_strategy(vf, env.get_current_state())
            next_vf = vf.get(best_action)

        val = vf.get(action)
        val += alpha * (rw + gamma * next_vf - val)

        vf.put(action, val)

    #print("number of actions = " + str(actions_number))
    return actions_number


def td_learning(td_algorithm, goal_coordinates, env, trials_number=250, action_default=0.5):
    target = env.get_state_by_coords(goal_coordinates)
    action_numbers = []
    i = 0

    vf = sln.ValueFunction()
    for v in env.vertices():
        vf.add_if_absent(v.get_outcoming(), action_default + np.random.rand() * 0.001)

    env.reset()
    while True:
        #print("----------------------------------------------------")
        # Нужна проверка для доступности пути.
        if not env.has_path_to(target):
            env.reset()
            continue

        i += 1
        an = td_algorithm(env, vf, target)
        action_numbers.append(an)

        if not i % trials_number:
            #if sln.ln.exit_condition():
            break

        env.reset()
        #print("STEP:" + str(i))
        #print("----------------------------------------------------")

    return action_numbers


class PeriodicGreedyStrategy(object):
    def __init__(self, epsilon, tau_off=1):
        self.tau_off = tau_off
        self.epsilon = epsilon
        self.last_time = {}
        self.t = 0

    def reset(self):
        self.last_time.clear()
        self.t = 0

    def choose_action(self, vf, state):
        self.t += 1
        can_update = False

        actions = state.get_outcoming()
        allowed_actions = [act for act in actions if self.is_allowed(act)]
        is_greedy = np.random.rand() >= self.epsilon

        if is_greedy and allowed_actions:
            i = np.argmax(map(lambda x: vf.get(x), allowed_actions))
            choosen_act = allowed_actions[i]

        else:
            i = np.random.randint(0, len(actions))
            choosen_act = actions[i]

        if self.is_allowed(choosen_act):
            self.last_time[choosen_act] = self.t
            can_update = True

        return choosen_act, can_update

    def is_allowed(self, action):
        return self.t - self.last_time.get(action, -self.tau_off) >= self.tau_off


def periodic_Qlearning(env, vf, goal, strategy, alpha, gamma, reward, penalty, always_update=False):
    assert isinstance(env, Environment)
    assert isinstance(vf, sln.ValueFunction)
    assert isinstance(strategy, PeriodicGreedyStrategy)

    actions_number = 0
    strategy.reset()

    while env.get_current_state() is not goal:
        if actions_number >= MAX_ACTIONS_NUMBER:  # for testing torus environment
            break

        action, can_update = strategy.choose_action(vf, env.get_current_state())

        index = env.get_current_state().get_outcoming().index(action)
        action_success = action.is_available()  # Она находиться здесь тк доступности разыгрываются для нового состояния
        # После действия либо не рызгрываются вообще, поэтому мы знвем успех
        # действия заранее. Если это измениться, вдруг нужно будет менять и здесь
        env.update_state(index)
        actions_number += 1
        next_vf = 0.0
        rw = 0.0

        if env.get_current_state() is goal:
            rw = reward
        else:
            if not action_success:
                rw = penalty
            best_action = greedy_strategy(vf, env.get_current_state())
            next_vf = vf.get(best_action)

        if can_update or always_update:
            val = vf.get(action)
            val += alpha * (rw + gamma * next_vf - val)
            vf.put(action, val)

    # print("number of actions = " + str(actions_number))
    return actions_number
