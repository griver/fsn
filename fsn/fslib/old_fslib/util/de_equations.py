# equations & equations factories which describes dynamics of functional systems
import math as m
import numpy as np
from ..fs.functional_system import BaseMotor, BaseSecondary, BaseMotivational


#-------utility eqs--------------------------------------------
def f_maker(k, threshold):
    def f(x):
        #print("WASSUP??")
        tmp = (-k) * (x - threshold)
        return 1/(1 + np.exp(tmp))
    return f


def discrete_f_maker(threshold):

    def f(x):
        return x >= threshold

    return f


def h_maker2(func):
    def h(x):
        return 1 - func(x)

    return h


def h_maker(k, threshold):
    f = f_maker(k, threshold)
    def h(x):
        return 1 - f(x)
    return h


def read_value_maker(text):
    def read_func(*args):
        str_val = input(text)
        return float(str_val)
    return read_func


def foo_maker(start, end):
    def func(*args):
        func.counter += 1
        if(func.counter == start):
            print("Start")
        return start <= func.counter <= end
    func.counter = 0
    return func


#------/utility eqs--------------------------------------------


#-------equation #1--------------------------------------------
def delta_si_maker(k, x0, sigma1):
    #f_x0 = f_maker(k, x0)
    f_x0 = discrete_f_maker(x0)

    def calc_delta_si(si, ri, input_sum):
        delta_si = si*(f_x0(ri) - input_sum -si) #+ sigma1 * random.random()
        return delta_si

    return calc_delta_si


def delta_ri_maker(taui, sigma2):

    def calc_delta_ri(si, ri, ii):
        delta_ri = ri*(si + ii - ri) #+ sigma2 * random.random()
        delta_ri = delta_ri/taui

        return delta_ri

    return calc_delta_ri
#-------/equations #1--------------------------------------------

#-------equation #3--------------------------------------------
def tauc_maker(tauc0, tauc1, f_x1):

    def tauc(si):
        result = tauc0 + tauc1* f_x1(si)
        return result

    return tauc

def delta_ci_maker(k, x1, tauc0, tauc1):
    f_x2 = f_maker(k, x1)
    f_x1 = discrete_f_maker(x1)
    tauc = tauc_maker(tauc0, tauc1, f_x1) #lambda x: 1

    def calc_delta_ci(si, ci):
        delta_ci = (f_x1(si)-ci)/tauc(si)
        return delta_ci

    return calc_delta_ci

#-------/equation #3------------------------------------------------------


#-------equation #10 and #4------------------------------------------------
"""def ii_maker(alpha, beta, gamma, is_motor):

    if is_motor:
        def calc_ii(ia, ar, ci, sec_influence):
            ii = alpha*ia - beta*ar - gamma*ci + sec_influence
            return ii
    else:
        def calc_ii(ia, ar, ci):
            ii = alpha*ia - beta*ar - gamma*ci
            return ii

    return calc_ii  # """

def ii_maker(alpha, beta, gamma, is_motor):

    if is_motor:
        def calc_ii(fs_motor):
            sec_influence = fs_motor._calc_influence(lambda e:  isinstance(e.get_src(), BaseSecondary))
            ii = alpha*fs_motor._newIA - beta*fs_motor._newAR - gamma*fs_motor._C + sec_influence
            return ii
    else:
        def calc_ii(fs):
            ii = alpha*fs._newIA - beta*fs._newAR - gamma*fs._C
            return ii

    return calc_ii

#------/equation #10 and #4------------------------------------------------

#-------equation #7-------------------------------------------------------
def motor_ia_maker2(threshold_func, j):
    f = threshold_func
    h = h_maker2(f)
    start_val = 1 - j

    def calc_motor_ia(st, ei):
        motor_ia = f(st)*h((ei - start_val)**2)
        return motor_ia

    return calc_motor_ia

def motor_ia_maker3(threshold_func, index, coord_val):
    f = threshold_func
    h = h_maker2(f)
    start_val = 1 - coord_val

    def calc_motor_ia(motor_fs):
        #assert isinstance(motor_fs, BaseMotor)
        curr_val = motor_fs._env.get_current_state().coords()[index]
        motor_ia = f(motor_fs.is_motivated())*h((curr_val - start_val)**2)
        return motor_ia

    return calc_motor_ia

def motor_ia_maker4(threshold_func, index):
    f = threshold_func
    h = h_maker2(f)

    def calc_motor_ia(motor_fs):
        no_edge = index >= len(motor_fs._env.get_current_state().get_outcoming())
        motor_ia = f(motor_fs.is_motivated())*h(no_edge)
        return motor_ia

    return calc_motor_ia


def motor_ia_maker(k, threshold, j):
    f = f_maker(k, threshold)
    return motor_ia_maker2(f, j)


def motor_ar_maker2(threshold_func, j):
    h = h_maker2(threshold_func)
    end_val = j

    #def calc_motor_ar(ei):
    #    motor_ar = h((ei - end_val)**2)
    #    return motor_ar
    def calc_motor_ar(motor_fs):
        current = motor_fs._env.get_current_state().coords()[motor_fs._index]
        motor_ar = h((current - end_val)**2)
        return motor_ar

    return calc_motor_ar


def motor_ar_maker(k, threshold, j):
    f = f_maker(k, threshold)
    return motor_ar_maker2(f, j)
#------/equation #7-------------------------------------------------


#-------equation #8-------------------------------------------------
def sec_ia_maker(k, threshold):
    f = f_maker(k, threshold)
    h = h_maker2(f)

    def calc_ia(st, states_dist):
        ia = f(st) * h(states_dist)
        return ia

    return calc_ia


def sec_ar_maker(k, threshold):
    h = h_maker(k, threshold)

    def calc_ar(states_dist):
        ar = h(states_dist)
        return ar

    return calc_ar
#------/equation #8-------------------------------------------------

#-------motiv AR-------------------------------------------------
def mot_ar_maker(base_tau, decrease_tau):

    def calc_ar(fs):
        assert isinstance(fs, BaseMotivational)
        curr = fs._env.get_current_state()
        if fs._goal_state == fs._env.get_current_state():
            return 1.0
        else:
            return 0.0

    return calc_ar
#------/motiv AR--------------------------------------------------


def delta_RS_rk4(h=1.0):

    def calc_delta_RS_rk4(fs):   # , h, env_interaction, influence_sum):
            influence_sum = fs._calc_influence(lambda e: isinstance(e.get_src(), BaseMotor))
            env_interaction = fs._newI

            K_1R = h * fs._delta_ri(fs._S, fs._R, env_interaction)
            K_1S = h * fs._delta_si(fs._S, fs._R, influence_sum)

            K_2R = h * fs._delta_ri(fs._S + 0.5 * K_1S, fs._R + 0.5 * K_1R, env_interaction)
            K_2S = h * fs._delta_si(fs._S + 0.5 * K_1S, fs._R + 0.5 * K_1R, influence_sum)

            K_3R = h * fs._delta_ri(fs._S + 0.5 * K_2S, fs._R + 0.5 * K_2R, env_interaction)
            K_3S = h * fs._delta_si(fs._S + 0.5 * K_2S, fs._R + 0.5 * K_2R, influence_sum)

            K_4R = h * fs._delta_ri(fs._S + K_3S, fs._R + K_3R, env_interaction)
            K_4S = h * fs._delta_si(fs._S + K_3S, fs._R + K_3R, influence_sum)

            delta_R = (K_1R + (2 * K_2R) + (2 * K_3R) + K_4R)/6
            delta_S = (K_1S + (2 * K_2S) + (2 * K_3S) + K_4S)/6

            return delta_R, delta_S

    return calc_delta_RS_rk4


def delta_RS_with_noise(numerical_integration, noise=0.001, IA_level=0.02):

    def calc_delta_RS(fs):
        delta_R, delta_S = numerical_integration(fs)
        delta_R += noise * np.random.rand() * (fs._newIA > IA_level)
        delta_S += noise * np.random.rand() * (fs._newIA > IA_level)

        return delta_R, delta_S

    return calc_delta_RS

def _calc_rk4_c(fs, h):
        K_1C = h * fs._delta_ci(fs._S, fs._C)
        K_2C = h * fs._delta_ci(fs._S, fs._C + 0.5 * K_1C)
        K_3C = h * fs._delta_ci(fs._S, fs._C + 0.5 * K_2C)
        K_4C = h * fs._delta_ci(fs._S, fs._C + K_3C)
        delta_C = (K_1C + (2 * K_2C) + (2 * K_3C) + K_4C)/6
        return delta_C