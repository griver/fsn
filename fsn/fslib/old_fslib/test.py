# coding=utf-8

from .util import fs_builder as FSBuilder
from .util import env_builder as EnvBuilder
from .learning import learning as ln

from .util.plots import PlotBuilder


def show_curves(x_axis, func, *funcs):
    pb = PlotBuilder()
    pb.create_figure(1, 1)
    pb.plot_curves(0, x_axis, func, *funcs)
    pb.show()


def show_bars():
    pb = PlotBuilder()
    pb.create_figure(1, 1)
    bars = []
    bars.append(([ (50, 50), (150, 10) ], "first") )
    bars.append(([ (10, 50), (100, 20), (130, 10) ], "second"))
    pb.plot_bars(0, (0,200), (0, 40), 10, *bars)
    pb.show()


def line(n):
    env = EnvBuilder.line() #starting point (0)
    target = env.get_vertex(1) #point (1)
    net = FSBuilder.create_empty_network()
    motiv = FSBuilder.simple_motiv(env, target)
    act = FSBuilder.motor(env, net, 0, 1)
    #act = FSBuilder.lm_secondary2(env, motiv, env.get_vertex(0), target)
    net.add_motiv(motiv)
    net.add_motor(act)
    s = []
    r = []
    c = []
    ii = []
    ia = []
    ar = []

    for i in xrange(0, n):
        if i == 140:
            pass
        net.recalc()
        net.apply()
        s.append(act.get_S())
        r.append(act.get_R())
        c.append(act.get_C())  # * FSBuilder.act_gamma)
        ii.append(act._I)
        ia.append(act.get_IA())  # * FSBuilder.act_alpha)

    pb = PlotBuilder()
    pb.create_figure(1, 1)
    pb.plot_curves(0, range(0, n), (s, '-', 'S'), (r, '-', 'R'), (c, ':', 'C'), (ia, ':', 'IA'))
    pb.show()
    #pl.savefig('single act_fs.png')


def base(n):
    env = EnvBuilder.direct_square()
    target = env.get_vertex(3)
    net = FSBuilder.create_empty_network()
    #motiv = FSBuilder.simple_motiv(env, target)
    motiv = FSBuilder.motiv(env, target)
    #fast_fs = FSBuilder.base_fs(env, motiv, 0, 1)
    fs = motiv
    net.add_motiv(motiv)
    #net.add_motor(fast_fs)

    s = []
    r = []
    ii = []
    ia = []
    ar = []

    for i in xrange(0, n):

        net.recalc()
        net.apply()
        s.append(fs.get_S())
        r.append(fs.get_R())
        ia.append(fs.get_IA())
        ar.append(fs.get_AR())
        ii.append(fs.get_I())
        #print("fast_fs.S = ", fast_fs.get_S())

    pb = PlotBuilder()
    pb.create_figure(1, 1)
    pb.plot_curves(0, range(0, n),
                    (s, '-', 'S'),
                    (r, '-', 'R'),
                    (ia, '--', 'IA'),
                    (ar, '--', 'AR'),
                    (ii, '-', 'I'))
    pb.show()


def secondary(n):
    env = EnvBuilder.direct_square()
    target = env.get_vertex(3)
    net = FSBuilder.create_empty_network()
    motiv = FSBuilder.simple_motiv(env, target)
    #fast_fs = FSBuilder.base_fs(env, motiv, 0, 1)

    fs = FSBuilder.secondary(env, motiv, env.get_vertex(0), env.get_vertex(1))
    net.add_motiv(motiv)
    net.add_vertex(fs)

    s = []
    r = []
    ia = []
    ar = []

    for i in xrange(0, n):
        net.recalc()
        net.apply()
        s.append(fs.get_S())
        r.append(fs.get_R())
        ia.append(fs.get_IA())
        ar.append(fs.get_AR())
        #print("fast_fs.S = ", fast_fs.get_S())

    pb = PlotBuilder()
    pb.create_figure(1, 1)
    pb.plot_curves(0, range(0, n), (s, 'b-', 'S'), (r, 'k-', 'R'), (ia, 'g--', 'IA'), (ar, 'r--', 'AR'))
    pb.show()


def fight(n, k = None):
    if k is None:
        k = int(input("Введите число конкурирующих систем: "))

    env = EnvBuilder.line()
    target = env.get_vertex(1)
    net = FSBuilder.create_empty_network()
    motiv = FSBuilder.simple_motiv(env, target)
    net.add_motiv(motiv)

    acts  = []
    s = []
    lines = []
    #net.create_cnet("MOTOR", 1.5)

    for i in xrange(0, k):
        act = FSBuilder.motor(env, net, 0, 1)
        #act = FSBuilder.lm_secondary(net, env, motiv, env.get_vertex(0), target)
        #act = FSBuilder.motiv(env, target)
        act.name = "act"+ str(i+1)
        #net.add_motor(act)
        net.add_in_cnet(act, "MOTOR")
        acts.append(act)
        s.append([])
        lines.append((s[i], '-', act.name))

    for  i in xrange(0, n):
        net.recalc()
        net.apply()
        for j in xrange(0, k):
            s[j].append(acts[j].get_S())

    show_curves(range(0,n), *lines)


def sec_influence_test(n, k = None):
    if k is None:
        k = int(input("Введите число конкурирующих систем: "))

    env = EnvBuilder.line()
    target = env.get_vertex(1)
    net = FSBuilder.create_empty_network("network_with_secondary_fs")
    motiv = FSBuilder.simple_motiv(env, target)
    net.add_motiv(motiv)
    #sec = FSBuilder.secondary(env, motiv, env.get_current_state(), target)
    sec = FSBuilder.lm_secondary(net, env, motiv, env.get_current_state(), target)
    sec.name = "secondary"
    sec._active_threshold = 2.0
    net.add_vertex(sec)

    acts  = []
    s = []
    sec_s = []
    lines = []
    for i in xrange(0, k):
        act = FSBuilder.motor(env, net, 0, 1)
        #act.active_threshold = 2.0
        act.name = "act"+ str(i+1)
        net.add_motor(act)
        acts.append(act)
        s.append([])
        lines.append((s[i], '--', act.name))

    for fs in net.all_motor():
        if fs is acts[0]:
            net.add_edge(sec, fs, 1.5)
        else:
            net.add_edge(sec, fs, -1.5)

    #net.write_to_file()
    for i in xrange(0, n):
        net.recalc()
        net.apply()
        for j in xrange(0, k):
            s[j].append(acts[j].get_S())
        sec_s.append(sec.get_S())

    tmp = net.get_action()

    show_curves(range(0, n), (sec_s, 'o--', sec.name), *lines)
    #return tmp[0] == acts[0]


def secondary_influence(n, iter, system_number):
    win = 0
    loose = 0
    for i in xrange(0, n):
        if sec_influence_test(iter, system_number):
            win += 1
        else:
            loose += 1
    print("Побед: " + str(win))
    print("Пордажений: " + str(loose))


def single_learning(goal_coordinates, env=EnvBuilder.cube()):

    sec_cnet_weight = 1.0
    sec_motiv_weight = 1.5
    net = FSBuilder.create_edge_moving_network(env, FSBuilder.edge_moving_motor, "base_network")

    assert isinstance(env, ln.Environment)
    assert isinstance(net, ln.BaseFSNetwork)

    target = env.get_state_by_coords(goal_coordinates)
    motiv = FSBuilder.simple_motiv(env, target)
    net.add_motiv(motiv)

    logger = ln.TrialLogger()

    i = 0
    while True:
        print("----------------------------------------------------")
        ln.trial(net, env, logger)
        ln.network_update(net, env, logger, sec_cnet_weight, sec_motiv_weight)

        i += 1
        #print(i)
        #if i > 30:
        if ln.exit_condition(): break
        ln.draw_trial(net, env, logger)

        ln.reset(net, env, logger)
        print("----------------------------------------------------")
    #net.write_to_file("learned_graph.dot")
    return net


def multi_learning(goals_coordinates, env=EnvBuilder.slingshot()):
    sec_cnet_weight = 1.0
    sec_motiv_weight = 1.5
    net = FSBuilder.create_network(env, FSBuilder.motor, "base_network")

    assert isinstance(env, ln.Environment)
    assert isinstance(net, ln.BaseFSNetwork)

    for t in goals_coordinates:
        target = env.get_state_by_coords(t)
        motiv = FSBuilder.motiv(env, target)
        net.add_motiv(motiv)

    logger = ln.TrialLogger()
    i = 0
    while True:
        i+=1
        print("----------------------------------------------------")
        ln.trial(net, env, logger)
        ln.network_update(net, env, logger, sec_cnet_weight, sec_motiv_weight)
        ln.reset(net)
        if i > 30 :
            names = map(lambda x: x.name, net.all_secondary())
            names.sort()
            for name in names:
                print(name)
        #   if not i % 5:
            #ln.draw_trial(net, env, logger)
            ln.draw_trial_bars(net, env, logger)
            if ln.exit_condition(): break
        print("----------------------------------------------------")


def stochastic_fork_test(n=1000):
    env, e1, e2 = EnvBuilder.fork()
    l1 = []
    l2 = []
    for i in xrange(0, n):
        env.reset()
        e1.is_available()
        l1.append(e1.is_available())
        l2.append(e2.is_available())
        if l1[i] + l2[i] != 1:
            print("Все работает неправильно!")

    s1 = reduce(lambda x, y: x + y, l1, 0)
    s2 = reduce(lambda x, y: x + y, l2, 0)
    print(s1)
    print(s2)
    print(s1+s2)

#--------------actor-critic trash tests---------------------------------------------
import numpy as np

def logg(step, result):
        if step:
            print("go th the right")
        else:
            print("go th the left")

        if result == False:
            print("FAIl :(")
        else:
            print("WIN :(")

def get_list(n, p):
    return np.random.randint(1, 11, n) <= p

def direct_actor_test(e, r, ml, mr, array):
    is_right = False
    acts = [0.0,0.0]
    all_acts = 0.0
    log = []
    log.append([])
    log.append([])

    st = []
    for i in xrange(0, len(array)):
        if mr == ml:
            is_right = np.random.rand() > 0.5
        else:
            is_right =  mr > ml

        st.append(is_right)


        all_acts += 1.0
        q = (is_right == array[i])
        acts[is_right] += 1.0
        mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
        ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
        #log[0].append(ml)
        #log[1].append(mr)
        r += 1/all_acts * (q - r)
        #logg(is_right, q)

        if q == False:
            #print("Then:")
            is_right = not is_right

            all_acts += 1.0
            q = (is_right == array[i])
            acts[is_right] += 1.0
            mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
            ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
         #   log[0].append(ml)
         #   log[1].append(mr)
            r += 1/all_acts * (q - r)
            #logg(is_right, q)
        log[0].append(ml)
        log[1].append(mr)

        #print("acts:" + str(acts))
        #print("r=" + str(r) +  "  ml=" + str(ml) + "  mr=" + str(mr))
        #print("------------------------------------------------------")

    show_curves(range(0,len(log[0])),
              (log[0], 'r-', "left"),
              (log[1], 'b-', "right")
    )

    right = sum(st) / float(len(st))
    print("right first probability: " + str(right))
    print("left first probability: " + str(1.0 - right))

def actor_test(e, r, ml, mr, array):
    is_right = False
    acts = [0.0,0.0]
    all_acts = 0.0
    log = []
    log.append([])
    log.append([])

    st = []
    for i in xrange(0, len(array)):
        if mr == ml:
            is_right = np.random.rand() > 0.5
        else:
            is_right =  mr > ml

        st.append(is_right)


        all_acts += 1.0
        q = (is_right == array[i])
        acts[is_right] += 1.0
        if is_right:
            mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
        else:
            ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
        #log[0].append(ml)
        #log[1].append(mr)
        r += 1/all_acts * (q - r)
        #logg(is_right, q)

        if q == False:
            #print("Then:")
            is_right = not is_right

            all_acts += 1.0
            q = (is_right == array[i])
            acts[is_right] += 1.0
            if is_right:
                mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
            else:
                ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
         #   log[0].append(ml)
         #   log[1].append(mr)
            r += 1/all_acts * (q - r)
            #logg(is_right, q)
        log[0].append(ml)
        log[1].append(mr)

        #print("acts:" + str(acts))
        #print("r=" + str(r) +  "  ml=" + str(ml) + "  mr=" + str(mr))
        #print("------------------------------------------------------")

    show_curves(range(0,len(log[0])),
              (log[0], 'r-', "left"),
              (log[1], 'b-', "right")
    )

    right = sum(st) / float(len(st))
    print("right first probability: " + str(right))
    print("left first probability: " + str(1.0 - right))

def da_and_weighted_average(alpha, e, r, ml, mr, array):
    is_right = False
    acts = [0.0,0.0]
    all_acts = 0.0
    log = []
    log.append([])
    log.append([])

    st = []
    for i in xrange(0, len(array)):
        if mr == ml:
            is_right = np.random.rand() > 0.5
        else:
            is_right =  mr > ml

        st.append(is_right)


        all_acts += 1.0
        q = (is_right == array[i])
        acts[is_right] += 1.0
        mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
        ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
        #log[0].append(ml)
        #log[1].append(mr)
        r += alpha * (q - r)
        #logg(is_right, q)

        if q == False:
            #print("Then:")
            is_right = not is_right

            all_acts += 1.0
            q = (is_right == array[i])
            acts[is_right] += 1.0
            mr = mr + e*(is_right - acts[1]/float(all_acts) ) * (q - r)
            ml = ml + e*((not is_right) - acts[0]/float(all_acts) ) * (q - r)
         #   log[0].append(ml)
         #   log[1].append(mr)
            r += alpha * (q - r)
            #logg(is_right, q)
        log[0].append(ml)
        log[1].append(mr)

        #print("acts:" + str(acts))
        #print("r=" + str(r) +  "  ml=" + str(ml) + "  mr=" + str(mr))
        #print("------------------------------------------------------")

    show_curves(range(0,len(log[0])),
              (log[0], 'r-', "left"),
              (log[1], 'b-', "right")
    )

    right = sum(st) / float(len(st))
    print("right first probability: " + str(right))
    print("left first probability: " + str(1.0 - right))
#--------------/trash---------------------------------------------


import random as rnd

class Test(object):
    choice = None
    updated = False
    count = 0

    @staticmethod
    def c_recalc(instance):
        if Test.updated: return
        Test.updated = True


        Test.choice = rnd.choice(xrange(Test.count))

    @staticmethod
    def c_apply(instance):
        Test.updated = False
        if instance.id == Test.choice:
            instance._is_choosen = True
        else:
            instance._is_choosen = False



    def __init__(self):
        self.id = Test.count
        self._is_choosen = False
        Test.count += 1

    def is_choosen(self):
        return self._is_choosen

    def recalc(self):
        Test.c_recalc(self)

    def apply(self):
        Test.c_apply(self)


if __name__ == "__main__":

    l = []
    for i in xrange(5):
        l.append(Test())

    for i in range(15):
        map(lambda x: x.recalc(), l)
        map(lambda x: x.apply(), l)
        print([i.id for i in l if i.is_choosen()])


    exit()
    #stochastic_fork_test(10000)
    print("Hello Bred!")
    torus = EnvBuilder.torus(3,3)
    torus.reset()
    torus.write_to_file("torus")

    print EnvBuilder.random_array(["Bob", "Alice", "Mike"], [0.25, 0.25, 0.5], 10)
    #line(200)
    #base(300)
    #fight(800, 4)
    #secondary(200)
    single_learning((1, 1, 1), EnvBuilder.rhomb())
    #multi_learning([(0, 0, 0), (0, 0, 1), (1, 0, 0)])
    #sec_influence_test(200, 8)
    #line(200)
    """l = []
    l.extend(get_list(10000, 9))
    #l.extend(get_list(5000, 1))
    #l.extend(get_list(10000, 9))
    #l.extend(get_list(10000, 1))
    direct_actor_test(0.05, 1.0 ,1.0, 1.0, l)
    #da_and_weighted_average(0.01, 0.05, 1.0 ,1.0, 1.0, l)
    da_and_weighted_average(0.01, 0.5, 20.0 ,20.0, 1.0, l)
    # epsilon - скорость реакции на не верные цености действий
    # alpha - так же скорость реакции но меньше.
    #    если уменьшать то увеличивается доля выбора более выгодного действия
    #    но если увеличивать то доля стремиться к реальному соотношению вероятностей
    da_and_weighted_average(0.005, 0.5, 1.0 ,20.0, 1.0, l)
    #actor_test(0.001, 1.0 ,1.0, 1.0, l)"""

else:
    pass

