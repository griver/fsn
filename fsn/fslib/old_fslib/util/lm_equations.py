import math as m

from ..fs.lmimpl import LMSecondary
from ..fs import BaseSecondary, CompetitiveNetwork
from ..env import Environment
import random
import numpy as np

MIN_IA = 0.98

def sec_IA_maker(ia_value = MIN_IA):

    def calc_ia(fs):
        assert isinstance(fs, LMSecondary)
        assert isinstance(fs._env, Environment)
        env = fs._env
        edge = next((e for e in fs.get_incoming() if e.get_src() == fs._motiv_fs), None)
        if fs.IA_point() == env.get_current_state() and fs._motiv_fs.is_active():
            return ia_value + 0.0002 * random.random()
        else:
            return 0.0

    return calc_ia

def weighted_IA_maker(ia_value = MIN_IA):

    def calc_ia(fs):
        assert isinstance(fs, LMSecondary)
        assert isinstance(fs._env, Environment)
        env = fs._env
        edge = next((e for e in fs.get_incoming() if e.get_src() == fs._motiv_fs), None)
        if fs.IA_point() == env.get_current_state() and fs._motiv_fs.is_active():

            out_infl = lambda e: isinstance(e.get_src(),BaseSecondary) and e.get_src().IA_point() != fs.IA_point()
            inc2 = filter(out_infl, fs.get_incoming())
            mean2 = reduce(lambda y, x: x.get_src().get_S()*x.weight() + y, inc2, 0.0)
            if len(inc2):
                mean2 /= len(inc2)


            return ia_value * edge.weight() + 0.0002 * random.random() +  mean2*0.02
        else:
            return 0.0

    return calc_ia



def sec_AR_maker(ia_value = MIN_IA):

    def calc_ar(fs):
        return 0

    return calc_ar

def sec_ii_maker( threshold):

    def calc_ii(fs):
        assert isinstance(fs, LMSecondary)
        sum = 0
        n = 0
        cnet = fs.get_cnet()
        if cnet is not None:
            for v in fs.get_cnet().vertices:
                if v.get_S() > threshold:
                    sum += v.get_S()
                    n += 1

        mean = sum

        return (n * fs.get_S() >= mean) * fs._newIA

    return calc_ii

def sec_ii_maker2( threshold):

    def calc_ii(fs):
        assert isinstance(fs, LMSecondary)

        sum = 0
        n = 0
        cnet = fs.get_cnet()
        if cnet is not None:
            for v in fs.get_cnet().vertices:
                if v.get_S() > threshold:
                    sum += v.get_S()
                    n += 1

        mean = sum

        out_infl = lambda e: isinstance(e.get_src(),BaseSecondary) and e.get_src().IA_point() != fs.IA_point()
        inc2 = filter(out_infl, fs.get_incoming())
        mean2 = reduce(lambda y, x: x.get_src().get_S()*x.weight() + y, inc2, 0.0)
        if len(inc2):
            mean2 /= len(inc2)

        return (n * fs.get_S() >= mean) * (fs._newIA + mean2*0.02)

    return calc_ii


def delta_si(fs):
    assert isinstance(fs, LMSecondary)
    return fs._newI/2.0 - fs.get_S()/2.0

def delta_si_with_deactivation(fs):
    assert isinstance(fs, LMSecondary)
    return fs._newI/2.0 * (1 - fs._deactivated) - fs.get_S()/2.0