# coding=utf-8
from ..fs import BaseMotor, BaseSecondary, BaseMotivational
from ..util import fs_builder as FSBuilder
from ..util.plots import PlotBuilder
from ..fs import BaseFSNetwork
from ..env import Environment
from .logger import TrialLogger



def _find_predicate(sec, start, end, motiv):
    assert isinstance(sec, BaseSecondary)
    return sec.IA_point() is start and sec.AR_point() is end and sec.get_motivation() is motiv


def _find_unlearned_transit(path, trial_start, net, motiv):
    goal = motiv.get_goal()
    if goal is not path[-1]:
        raise ValueError("path must end with a goal state")

    secs = net.all_secondary()
    rng = range(0,len(path) - 1)
    rng.reverse()
    for i in rng:  # without last state, because last state is a goal state
        fs = next((s for s in secs if _find_predicate(s, path[i], path[i+1], motiv)), None)
        if fs is None:
            return i

        if i == trial_start:
            break

    return None


def single_goal_learning(env, net, motiv, sec_to_sec_weight, sec_to_motor_weight = 1.5):
    target = motiv.get_goal()
    motor_list = net.all_motor()

    activefs = 0
    n = 0
    st_path = []
    act_path = []
    prev_fs = None
    st_path.append(env.get_current_state())

    logger = []
    state_changes = []
    for v in net.vertices():
        logger.append([])

    while env.get_current_state() is not target: #and n < 500:
        n += 1
        #if n % 100 == 0:
        #    print(n)
        net.recalc()
        net.apply()
        active_motor_fs = filter(lambda fs: fs.is_active(), motor_list)

        for i in xrange(0, len(net.vertices())):
            logger[i].append(net.get_vertex(i).get_S())

        if len(active_motor_fs) > 1:
            raise ValueError("Одновременно активны несколько FS действия")
        elif len(active_motor_fs) == 0:
            prev_fs = None
            state_changes.append(env.get_current_state().get_id())
            continue

        curr_fs = active_motor_fs[0]
        if curr_fs is not prev_fs:
            print(curr_fs.name + " is active")
            activefs += 1
            prev_fs = curr_fs

        env.update_state(curr_fs.edge_index())
        state_changes.append(env.get_current_state().get_id())

        if env.get_current_state() is not st_path[-1]:
            st_path.append(env.get_current_state())
            act_path.append(curr_fs)

    #state_changes.append(target.get_id())
    #----show graphs-------------------
    print("active_fs == " + str(activefs))
    act_funcs = []
    sec_funcs = []
    for i in xrange(0, len(logger)):
        fs = net.get_vertex(i)
        if isinstance(fs, BaseMotor):
            act_funcs.append((logger[i], '-', fs.name))
        elif isinstance(fs, BaseSecondary):
            sec_funcs.append((logger[i], '--', fs.name))
    pb = PlotBuilder()
    if len(sec_funcs) > 0:
        pb.create_figure(3, 1)
        pb.plot_curves(1, range(0, n), *act_funcs)
        pb.plot_curves(2, range(0, n), *sec_funcs)
    else:
        pb.create_figure(2, 1)
        pb.plot_curves(1, range(0, n), *act_funcs)
    pb.plot_curves(0, range(0, n), (state_changes, 'o-', "State"))
    pb.show()
    #---/show graphs-------------------

    id = _find_unlearned_transit(st_path, net, motiv)
    if id is None:
        print("Весь путь от начального состояния до цели был выучен")
        return n, st_path, logger

    print("Добавляем вторичную фс для перехода из " + st_path[id].name + " в " + st_path[id + 1].name)
    print("она будет стимулировать активность системы  " + act_path[id].name)
    sec = FSBuilder.lm_secondary(env, motiv, st_path[id], st_path[id + 1])

    # find other secondary fast_fs associated with this state
    secs = filter(lambda fs: fs.IA_point() is st_path[id], net.all_secondary())

    net.add_vertex(sec)

    if len(secs) == 1:
        cnet_name = "CNET" + st_path[id].name
        net.create_cnet(cnet_name, sec_to_sec_weight)
        print("create competitive network: " + cnet_name)
        net.add_in_cnet(sec, cnet_name)
        net.add_in_cnet(secs[0], cnet_name)
    elif len(secs) > 1:
        cnet_name = secs[0].get_cnet_name()
        net.add_in_cnet(sec, cnet_name)

    net.add_vertex(sec)
    for fs in motor_list:
        if fs is act_path[id]:
            net.add_edge(sec, fs, sec_to_motor_weight)
        else:
            net.add_edge(sec, fs, -sec_to_motor_weight)
    return n, st_path, logger


def trial_stop_condition(env, net):
    """
    :return: True - если мы находимся в целевом состоянии хотя бы одной из активных мотивационных ФС.
    """
    ##assert isinstance(net, BaseFSNetwork)
    ##assert isinstance(env, Environment)
    return len(filter(lambda m: m.is_active() and env.get_current_state() is m.get_goal(),  net.all_motiv())) > 0


def trial(net, env, logger):
    """
    Функция получает агента(net), среду(env) и проводит испытание.
    Испытание заканчивается, когда агент достигает целевого состояния хотя бы одной из активных мотивационных ФС.
    Все действия агента записываются логгером.
    """
    #assert isinstance(net, BaseFSNetwork)
    #assert isinstance(env, Environment)
    assert isinstance(logger, TrialLogger)
    prev_fs = None
    prev_state = env.get_current_state()
    logger.start_trial(env, net)

    while not trial_stop_condition(env, net):
        net.recalc()
        net.apply()
        curr_motor = net.get_action()

        if curr_motor is not None:
            env.update_state(curr_motor.edge_index())  # """

            if prev_state is not env.get_current_state():
                for fs in net.all_motor(): fs.reset()

        prev_state = env.get_current_state()
        prev_fs = curr_motor
        logger.add(net, env)

        if logger.get_actions_number() > 0 and not logger.get_actions_number() % 15000:
            print("actions number = {0}".format(logger.get_actions_number()))

        #if logger.get_actions_number() == 4000:
        #    break

    #print("active_fs == " + str(logger.get_actions_number()))


def network_update(net, env, logger, sec_cnet_weight=1.0, sec_motor_weight=1.5, deactivation_delay=20):
    """
    По результатам испытания функция добавляет новые вторичные фс
    """
    assert isinstance(net, BaseFSNetwork)
    # assert isinstance(env, Environment)
    # assert isinstance(logger, TrialLogger)

    trial_path = logger.get_path()
    trial_actions = logger.get_actions()
    motivs = filter(lambda m: m.is_active() and trial_path[-1] == m.get_goal(), net.all_motiv())

    if len(motivs) != 1:
        raise ValueError("Ошибка: Количество мотивационных систем удовлеторивших потребность: " + str(len(motivs)))

    id = _find_unlearned_transit(trial_path, logger.get_last_start(), net, motivs[0])  #

    if id is None:
        #print("Весь путь от начального состояния до цели был выучен")
        return

    #print("Добавляем вторичную фс для перехода из " + trial_path[id].name + " в " + trial_path[id + 1].name)
    #print("она будет стимулировать активность моторной фс  " + trial_actions[id].name)
    sec = FSBuilder.lm_secondary2(net, env, motivs[0], trial_path[id], trial_path[id + 1], deactivation_delay)

    # find other secondary fast_fs associated with this state
    secs = filter(lambda fs: fs.IA_point() is trial_path[id], net.all_secondary())

    net.add_vertex(sec)

    # если существуют другие вторичные ФС связанные с тем же состоянием,
    # добавляем взаимноподовляющие связи между ними и новой вторичной ФС
    if len(secs) == 1:
        cnet_name = "CNET" + trial_path[id].name
        net.create_cnet(cnet_name, sec_cnet_weight)
        #print("create competitive network: " + cnet_name) #competitive network
        net.add_in_cnet(sec, cnet_name)
        net.add_in_cnet(secs[0], cnet_name)
    elif len(secs) > 1:
        cnet_name = secs[0].get_cnet_name()
        net.add_in_cnet(sec, cnet_name)

    # добавляем связи от вторичной фс к моторным фс
    for fs in net.all_motor():
        if fs is trial_actions[id]:
            net.add_edge(sec, fs, sec_motor_weight)
        else:
            net.add_edge(sec, fs, -sec_motor_weight)
    net.add_edge(motivs[0], sec, 0.98)

    # ищем существует ли в сети обратная к новой вторичная система
    contrs = filter(lambda fs: fs.IA_point() is trial_path[id + 1]
                        and fs.AR_point() is trial_path[id]
                        and fs.get_motivation() is motivs[0], net.all_secondary())

    if len(contrs) == 1:  # we add inhibitory connection between secondary systems which do the contrary actions.
        net.add_edge(sec, contrs[0], -sec_cnet_weight)
        net.add_edge(contrs[0], sec, -sec_cnet_weight)


def draw_trial(net, env, logger):
    assert isinstance(net, BaseFSNetwork)
    assert isinstance(logger, TrialLogger)
    if logger.get_count() == 0:
        print("Trial log is empty")
        return

    print("path length: " + str(len(logger.get_path()) - 1))
    motiv_funcs = []
    motor_funcs = []
    sec_funcs = []
    x_axis = range(0, logger.get_count())
    activities = logger.get_fs_activities()

    for i in xrange(0, len(activities)):
        fs = net.get_vertex(i)
        if isinstance(fs, BaseMotor):
            motor_funcs.append((activities[i], '-', fs.name))
        elif isinstance(fs, BaseSecondary):
            sec_funcs.append((activities[i], '--', fs.name))
        elif isinstance(fs, BaseMotivational):
            motiv_funcs.append((activities[i], '--', fs.name))

    pb = PlotBuilder()
    if len(sec_funcs) > 0 and logger.get_last_count() == 0:
        pb.create_figure(4, 1)
        pb.plot_curves(3, x_axis, *sec_funcs)
        pb.get_subplot_ax(3).set_ylim(-0.1, 1.1)
    else:
        pb.create_figure(3, 1)

    pb.plot_curves(2, x_axis, *motor_funcs)
    #pb.plot_curves(1, x_axis, *motiv_funcs)
    pb.plot_bars(1, (x_axis[0], x_axis[-1]), (0, 100), 0.98, *motiv_funcs)
    pb.plot_curves(0, x_axis, (logger.get_states(), 'o-', "State"))

    pb.show()


def draw_trial_bars(net, env, logger):
    assert isinstance(net, BaseFSNetwork)
    assert isinstance(logger, TrialLogger)
    if logger.get_count() == 0:
        print("Trial log is empty")
        return

    motiv_funcs = []
    motor_funcs = []
    sec_funcs = []
    x_axis = range(0, logger.get_count())
    activities = logger.get_fs_activities()

    for i in xrange(0, len(activities)):
        fs = net.get_vertex(i)
        if isinstance(fs, BaseMotor):
            motor_funcs.append((activities[i], '-', fs.name))
        elif isinstance(fs, BaseSecondary):
            sec_funcs.append((activities[i], '--', fs.name))
        elif isinstance(fs, BaseMotivational):
            motiv_funcs.append((activities[i], '--', fs.name))

    pb = PlotBuilder()
    if len(sec_funcs) > 0 and logger.get_last_count() == 0:
        pb.create_figure(4, 1)
        pb.plot_curves(3, x_axis, *sec_funcs)
        pb.get_subplot_ax(3).set_ylim(-0.1, 1.1)
    else:
        pb.create_figure(3, 1)

    pb.plot_bars(2, (x_axis[0], x_axis[-1]), (0, 400), 0.98, *motor_funcs)
    pb.plot_bars(1, (x_axis[0], x_axis[-1]), (0, 100), 0.98, *motiv_funcs)
    pb.plot_curves(0, x_axis, (logger.get_states(), 'o-', "State"))

    pb.show()


def reset(*resetables):  # обновит тех кого передадим!
    for obj in resetables:
        obj.reset()


def exit_condition():
    tmp = int(input("Хотите продолжить обучение? (1/ 0)\n"))
    return tmp == 0
