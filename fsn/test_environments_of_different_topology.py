# coding=utf-8
from test import *
from fslib.learning import discrete_fsn as dfsn
from fslib.learning import hierarchical_fsn as hfsn
#===== функции тестирующие алгоритмы на наборе сред ========================

def periodic_Qlearning_test(goal_coordinates, env,
                            tau_off, epsilon,
                            alpha, gamma=1.0, reward=1.0, penalty=0.0,
                            trials_number=250, always_update=False):
    strategy = tdl.PeriodicGreedyStrategy(epsilon, tau_off)
    algorithm = lambda en, vf, target: tdl.periodic_Qlearning(en, vf, target, strategy, alpha, gamma,
                                                              reward, penalty, always_update=always_update)
    return tdl.td_learning(algorithm, goal_coordinates, env, trials_number)


def test_periodicQ_and_save_results_to_a_folder(id2env, target_node_id, folder_to_save, runs_in_one_env,
                                                      trials_in_one_run):
    time_benchmarks = []
    PQ_01_01_same = [] # {alpha}_{epsilon}_{tau_off}
    PQ_01_01_inf = []  # {alpha}_{epsilon}_{tau_off}
    PQ_01_001_same = []  # {alpha}_{epsilon}_{tau_off}
    PQ_01_001_inf = []  # {alpha}_{epsilon}_{tau_off}

    for index, env in id2env.iteritems():
        target_node = env.get_vertex(target_node_id)

        print "##" * 25
        print "RUNS COMPLETED: #{0}".format(len(time_benchmarks))
        print "##" * 25

        for i in xrange(runs_in_one_env):
            env.reset()
            start_time = time.time()
            PQ01same = periodic_Qlearning_test(target_node.coords(), env, env_period,
                                                                0.1, alpha=0.1, gamma=0.9, penalty=0.0,
                                                                trials_number=trials_in_one_run)

            PQ01inf = periodic_Qlearning_test(target_node.coords(), env, 1000000,
                                                                0.1, alpha=0.1, gamma=0.9, penalty=0.0,
                                                                trials_number=trials_in_one_run)

            PQ001same = periodic_Qlearning_test(target_node.coords(), env, env_period,
                                             0.01, alpha=0.1, gamma=0.9, penalty=0.0,
                                             trials_number=trials_in_one_run)

            PQ001inf = periodic_Qlearning_test(target_node.coords(), env, 1000000,
                                            0.01, alpha=0.1, gamma=0.9, penalty=0.0,
                                            trials_number=trials_in_one_run)

            end_time = time.time()
            print 'All Periodic Qlearning time:', end_time - start_time
            time_benchmarks.append(end_time - start_time)

            PQ_01_01_same.append(PQ01same)
            PQ_01_01_inf.append(PQ01inf)

            PQ_01_001_same.append(PQ001same)
            PQ_01_001_inf.append(PQ001inf)

        save_json_data(folder_to_save + "/pq_(0.1,0.9,0.1,{0})_{1}.json".format('same',index),
                       PQ_01_01_same[len(PQ_01_01_same) - runs_in_one_env:])
        save_json_data(folder_to_save + "/pq_(0.1,0.9,0.1,{0})_{1}.json".format('inf',index),
                       PQ_01_01_inf[len(PQ_01_01_inf) - runs_in_one_env:])

        save_json_data(folder_to_save + "/pq_(0.1,0.9,0.01,{0})_{1}.json".format('same', index),
                        PQ_01_001_same[len(PQ_01_001_same) - runs_in_one_env:])
        save_json_data(folder_to_save + "/pq_(0.1,0.9,0.01,{0})_{1}.json".format('inf', index),
                        PQ_01_001_inf[len(PQ_01_001_inf) - runs_in_one_env:])

    print("average trial time in seconds: {0}".format(np.average(time_benchmarks)))


def conduct_test_on_envs_and_save_results_to_a_folder(id2env, target_node_id, folder_to_save, runs_in_one_env,
                                                      trials_in_one_run):
    time_benchmarks = []
    fs_paths = []
    fs_secondary = []
    q_1_9_1_paths = []  # q_alpha_gamma_epsilon
    q_1_9_01_paths = []
    q_01_99_01_paths = []
    rnd_paths = []



    for index, env in id2env.iteritems():
        target_node = env.get_vertex(target_node_id)

        print "##" * 25
        print "RUNS COMPLETED: #{0}".format(len(time_benchmarks))
        print "##" * 25

        for i in xrange(runs_in_one_env):
            env.reset()
            start_time = time.time()
            fs_path_len, fs_secondary_size = stoch_learning(target_node.coords(), env,
                                                            trials_number=trials_in_one_run)
            fsn_end_time = time.time()
            print 'FSN trial time:', fsn_end_time - start_time
            q_path_len_1_9_01 = Q_learning_test(target_node.coords(), env, 0.01, 0.1, 0.9,
                                                trials_number=trials_in_one_run, penalty=0.0)
            q_path_len_1_9_1 = Q_learning_test(target_node.coords(), env, 0.1, 0.1, 0.9,
                                               trials_number=trials_in_one_run, penalty=0.0)
            q_path_len_01_99_01 = Q_learning_test(target_node.coords(), env, 0.01, 0.01, 0.99,
                                                  trials_number=trials_in_one_run, penalty=0.0)



            q_end_time = time.time()
            print 'Average Q-learning time:', (q_end_time - fsn_end_time)/3.
            rnd_path_len = random_trials(target_node.coords(), env, trials_number=trials_in_one_run)

            end_time = time.time()
            print 'Random Search time:', end_time - q_end_time
            time_benchmarks.append(end_time - start_time)

            fs_paths.append(fs_path_len)
            fs_secondary.append(fs_secondary_size)
            q_1_9_1_paths.append(q_path_len_1_9_1)
            q_1_9_01_paths.append(q_path_len_1_9_01)
            q_01_99_01_paths.append(q_path_len_01_99_01)
            rnd_paths.append(rnd_path_len)

        save_json_data(folder_to_save + "/all_paths_{0}.json".format(index),
                       fs_paths[len(fs_paths) - runs_in_one_env:])
        save_json_data(folder_to_save + "/all_secondary_{0}.json".format(index),
                       fs_secondary[len(fs_secondary) - runs_in_one_env:])

        save_json_data(folder_to_save + "/q-learning_(0.1,0.9, 0.1)_{0}.json".format(index),
                        q_1_9_1_paths[len(q_1_9_1_paths) - runs_in_one_env:])
        save_json_data(folder_to_save + "/q-learning_(0.1,0.9, 0.01)_{0}.json".format(index),
                        q_1_9_01_paths[len(q_1_9_01_paths) - runs_in_one_env:])
        save_json_data(folder_to_save + "/q-learning_(0.01,0.99, 0.01)_{0}.json".format(index),
                        q_01_99_01_paths[len(q_01_99_01_paths) - runs_in_one_env:])

        save_json_data(folder_to_save + "/rnd_paths_{0}.json".format(index),
                       rnd_paths[len(rnd_paths) - runs_in_one_env:])

    print("average trial time in seconds: {0}".format(np.average(time_benchmarks)))


def test_discrete_fsn(id2env, target_node_id, folder_to_save,
                      runs_in_one_env, trials_in_one_run):
    #algorithm params:
    #lr = 0.05
    #fork_discount = 0.99
    #gamma = 1.
    T_off = 15000

    res2name_template = {}
    res2name_template['h_fsn_steps'] = folder_to_save + '/h_fsn_path({0},{1},{2})_{3}.json'
    res2name_template['h_fsn_size'] = folder_to_save + '/h_fsn_size({0},{1},{2})_{3}.json'

    res2name_template['h_qfsn_steps'] = folder_to_save + '/h_qfsn_path({0},{1},{2})_{3}.json'
    res2name_template['h_qfsn_size'] = folder_to_save + '/h_qfsn_size({0},{1},{2})_{3}.json'

    res2name_template['fsn_steps'] = folder_to_save + '/fsn_path({0},{1},{2})_{3}.json'
    res2name_template['fsn_size'] = folder_to_save + '/fsn_size({0},{1},{2})_{3}.json'

    res2name_template['pq'] = folder_to_save + '/pq_inf({0},{1},{2})_{3}.json'

    pq_params = {'T_off': 15000, 'alpha': 0.1, 'gamma': 0.9, 'epsilon': 0.01}  # 0.01, alpha=0.1, gamma=0.9
    # lr=0.05, fork_discount=0.95, gamma=0.99
    h_fsn_params = {'lr': 0.05, 'fork_discount': 0.99, 'gamma': 0.9} ## lr=0.01, fork_discount=0.95, gamma=0.99
    h_qfsn_params = {'lr': 0.05, 'fork_discount': 0.9, 'gamma': 0.9} # lr=0.05, fork_discount=0.9, gamma=0.9
    fsn_params = {'lr': 0.05, 'fork_discount': 0.99, 'gamma': 0.9}
    time_benchmarks = []

    for index, env in id2env.iteritems():
        target_node = env.get_vertex(target_node_id)
        print "##" * 25
        print "RUNS COMPLETED: #{0}".format(len(time_benchmarks))
        print "##" * 25

        h_fsn = {'net_size':[], 'num_steps':[]}
        h_qfsn = {'net_size': [], 'num_steps': []}
        fsn = {'net_size': [], 'num_steps': []}
        pq = []

        runs2num_steps = []
        runs2fsn_size = []


        for i in xrange(runs_in_one_env):
            start_time = time.time()
            num_steps, net_size = hfsn.train_hierarchical_fsn(target_node, env,
                                                        num_episodes=trials_in_one_run,
                                                        max_episode_steps=15000, lr=h_fsn_params['lr'],
                                                        fork_discount=h_fsn_params['fork_discount'],
                                                        gamma=h_fsn_params['gamma'], T_off=T_off, verbose=0,
                                                        use_Q_update=False)

            h_fsn['num_steps'].append(num_steps)
            h_fsn['net_size'].append(net_size)

            num_steps, net_size = hfsn.train_hierarchical_fsn(target_node, env,
                                                              num_episodes=trials_in_one_run,
                                                              max_episode_steps=15000, lr=h_qfsn_params['lr'],
                                                              fork_discount=h_qfsn_params['fork_discount'],
                                                              gamma=h_qfsn_params['gamma'], T_off=T_off, verbose=0,
                                                              use_Q_update=True)

            h_qfsn['num_steps'].append(num_steps)
            h_qfsn['net_size'].append(net_size)

            num_steps, net_size = hfsn.train_discrete_fsn(target_node, env,
                                                          num_episodes=trials_in_one_run,
                                                          max_episode_steps=15000, lr=fsn_params['lr'],
                                                          fork_discount=fsn_params['fork_discount'],
                                                          gamma=fsn_params['gamma'], T_off=T_off, verbose=0)

            fsn['num_steps'].append(num_steps)
            fsn['net_size'].append(net_size)

            num_steps = periodic_Qlearning_test(target_node.coords(), env, T_off,
                                                epsilon=pq_params['epsilon'], alpha=pq_params['alpha'],
                                                gamma=pq_params['gamma'], penalty=0.0,
                                                trials_number=trials_in_one_run)

            pq.append(num_steps)

            end_time = time.time()
            print 'DiscreteFSN FSN learning time:', end_time - start_time
            time_benchmarks.append(end_time - start_time)

        save_json_data(
            res2name_template['h_fsn_steps'].format(h_fsn_params['lr'], h_fsn_params['fork_discount'], h_fsn_params['gamma'], index),
            h_fsn['num_steps'])
        save_json_data(
            res2name_template['h_fsn_size'].format(h_fsn_params['lr'], h_fsn_params['fork_discount'], h_fsn_params['gamma'], index),
            h_fsn['net_size'])

        save_json_data(
            res2name_template['h_qfsn_steps'].format(h_qfsn_params['lr'], h_qfsn_params['fork_discount'], h_qfsn_params['gamma'], index),
            h_qfsn['num_steps'])
        save_json_data(
            res2name_template['h_qfsn_size'].format(h_qfsn_params['lr'], h_qfsn_params['fork_discount'], h_qfsn_params['gamma'], index),
            h_qfsn['net_size'])

        save_json_data(
            res2name_template['fsn_steps'].format(fsn_params['lr'], fsn_params['fork_discount'], fsn_params['gamma'], index),
            fsn['num_steps'])
        save_json_data(
            res2name_template['fsn_size'].format(fsn_params['lr'], fsn_params['fork_discount'], fsn_params['gamma'], index),
            fsn['net_size'])

        save_json_data(
            res2name_template['pq'].format(pq_params['alpha'], pq_params['gamma'], pq_params['epsilon'], index),
            pq)


    print("average trial time in seconds: {0}".format(np.mean(time_benchmarks)))

#===== /функции тестирующие алгоритмы на наборе сред ========================

def load_particular_envs_from_folder(env_indices, folder_with_env_files, env_type):
    """
    :return: a dict from indices to loaded environments
    """
    id2env = {}
    for i in env_indices:
        id2env[i] = EnvBuilder.load_stohastic_environment_from_dot(
            folder_with_env_files + "/{0}.dot".format(i), env_type)

    return id2env


def test_environments(folder_with_env_files, folder_to_save_results, env_type, trials_number):
    """
    loads environments from folder, tests algorithms on these environments and saves results in the specified folder
    """
    first_id = input("номер первой тестируемой среды:")
    last_id = input("номер последней тестируемой среды(включительно):")

    env_indices = range(first_id, last_id + 1)
    id2env = load_particular_envs_from_folder(env_indices, folder_with_env_files, env_type)

    full_save_path = folder_to_save_results + "/{0}".format(env_type.__name__.lower())
    target_node_id = id2env[env_indices[0]].get_vertices_number() / 2
    conduct_test_on_envs_and_save_results_to_a_folder(id2env, target_node_id, full_save_path, 10, trials_number)

    print "Тестирование алгоритмов на средах с {0} по {1} проведено".format(first_id, last_id)


def test_periodic_envs_in_torus(torus_x, torus_y, period, folder_to_save_results, trials_number, runs_per_env=10):
    print 'x=', torus_x, 'y=', torus_y, 'period=', period, 'runs=', runs_per_env, 'trials for one run=', trials_number
    first_id = input("номер первой тестируемой среды:")
    last_id = input("номер последней тестируемой среды(включительно):")

    env_indices = range(first_id, last_id + 1)
    create_periodic_env = lambda dims, start_state_id: altenv.PeriodicEnvironment(period=period, dimension=dims,
                                                                                  start_state_id=start_state_id)
    id2env = load_torus_envs(env_indices, torus_x, torus_y, None) #create_periodic_env)
    #full_save_path = folder_to_save_results + "/{0}x{1}/periodicenvironment_{2}".format(torus_x, torus_y, period)
    full_save_path = folder_to_save_results + "/{0}x{1}/stochasticenvironment".format(torus_x, torus_y)
    target_node_id = (torus_x / 2) * torus_y + torus_y / 2

    #conduct_test_on_envs_and_save_results_to_a_folder(id2env, target_node_id, full_save_path,
    #                                                  runs_per_env, trials_number)
    #print "Запускаем тестирование periodic Q learning'a: "
    #test_periodicQ_and_save_results_to_a_folder(id2env, target_node_id, full_save_path,
    #                                            runs_per_env, trials_number)
    test_discrete_fsn(id2env, target_node_id, full_save_path, runs_per_env, trials_number)


def load_torus_envs(env_indices, torus_x, torus_y, create_env):
    """
    :return: a dict from indices to loaded environments
    """
    id2env = {}
    for i in env_indices:
        probs = get_json_data("../env_data/torus_probs/{0}x{1}/{2}.json".format(torus_x, torus_y, i))
        get_probs = lambda n: probs[:n]
        id2env[i] = EnvBuilder.torus(torus_x, torus_y,
                                 env_class=create_env,
                                 get_probabilities=get_probs)

    return id2env


def test_periodic():
    create_periodic_env = lambda dims, start_state_id: altenv.PeriodicEnvironment(period=5, dimension=dims,
                                                                                  start_state_id=start_state_id)
    # env = EnvBuilder.load_stohastic_environment_from_dot("random_envs/barabasi_albert/N25_M3/1.dot",
    #                                               init_env=create_periodic_env)

    probs = get_json_data("../env_data/torus_probs/{0}x{1}/{2}.json".format(5, 5, 1))
    get_probs = lambda n: probs[:n]
    env = EnvBuilder.torus(5, 5, create_periodic_env, get_probs)

    for i in xrange(10):
        print '<<<<<<<<<<<<<<<<<< TRIAL {0} >>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(i)
        env.reset()
        for j in xrange(13):
            rand_action = rnd.randrange(len(env.get_current_state().get_outcoming()))

            print 'state: {0}'.format(env.get_current_state()), 'action: {0}'.format(rand_action)
            env.update_state(rand_action)


env_period = 15000


def check_sparse_grid(h,w, cell_side, probs_folder, n_rep=100):
    results = []
    for i in xrange(1,11):
        probs = get_json_data(probs_folder + '/{0}.json'.format(i))
        get_probs = lambda n: probs[:n]
        env = EnvBuilder.sparse_grid(h,w, get_probabilities=get_probs, cell_len=cell_side)
        target_node = env.get_vertex(-1)

        num_steps = random_trials(target_node.coords(), env, trials_number=n_rep)
        results.append(num_steps)
        #for _ in xrange(n_rep):
            #env.reset()
            #results.append(env.has_path_to(target_node))
    results =  np.array(results)
    return np.mean(results)


def test_sparse_grid(grid_size, cell_size, period, folder_to_save_results, trials_number, runs_per_env=10):
    h = w = grid_size
    print 'test sparse grid environment:'
    print 'h=', h, 'w=', w, 'period=', period, 'runs=', runs_per_env, 'trials for one run=', trials_number
    first_id = input("номер первой тестируемой среды:")
    last_id = input("номер последней тестируемой среды(включительно):")

    env_indices = range(first_id, last_id + 1)

    if period is None:
        full_save_path = folder_to_save_results + "/{0}x{1}x{2}/stochasticenvironment".format(h, w, cell_size)
        create_env = EnvBuilder.StochasticEnvironment
    else:
        full_save_path = folder_to_save_results + "/{0}x{1}x{2}/periodicenvironment_{3}".format(h,w,cell_size, period)
        create_env = lambda dims, start_state_id: altenv.PeriodicEnvironment(period=period, dimension=dims,
                                                                             start_state_id=start_state_id)

    id2env = load_sparse_grid_envs(env_indices, h, w, cell_size, create_env=create_env)
    target_node_id = id2env[first_id].get_vertices_number() - 1

    test_discrete_fsn(id2env, target_node_id, full_save_path, runs_per_env, trials_number)


def load_sparse_grid_envs(env_indices, h, w, cell_size, create_env):
    """
    :return: a dict from indices to loaded environments
    """
    id2env = {}
    for i in env_indices:
        probs = get_json_data("../env_data/torus_probs/{0}x{1}/{2}.json".format(10, 10, i))
        get_probs = lambda n: probs[:n]
        id2env[i] = EnvBuilder.sparse_grid(h, w, get_probabilities=get_probs,
                                 env_class=create_env, cell_len=cell_size)

    return id2env


if __name__ == "__main__":
    #test_sparse_grid(grid_size=4, cell_size=4, period=1,
    #                 folder_to_save_results='random_envs/sparse_grid', trials_number=1000)

    from matplotlib import pyplot as plt
    from itertools import product
    import  sys
    from tqdm import trange

    def mean(data):
        return np.mean(np.stack(data, axis=0), axis=0)

    def median(data):
        return np.median(np.stack(data, axis=0), axis=0)

    def last_10_trials_stats(data):
        data = np.stack(data, axis=0)
        mdn = np.median(data[:, -10:])
        avr = np.mean(data[:, -10:])
        return mdn, avr

    #for l in [1,2,4,8]:
    #    print 'Mean RND on: 4x4, cell_side={0} : '.format(l), check_sparse_grid(4,4, l, '../env_data/torus_probs/10x10', n_rep=500)

    #lr=0.01, fork_discount=0.95, gamma=0.99
    lrs = [ 0.05] #[0.01, 0.03, 0.05, 0.07, 0.1]
    fork_discounts = [0.99] #[.9, 0.99, 1.]  #[0.9, 0.95, 0.99, 1.]
    gammas = [0.9]  #[.9, 0.99, 1.] #[0.9, 0.95, 0.99, 1.]

    create_env= None
    trials_in_one_run = 1000

    create_periodic_env = lambda dims, start_state_id: altenv.PeriodicEnvironment(period=1, dimension=dims,
                                                                                  start_state_id=start_state_id)
    h, w = 4, 4
    probs = get_json_data("../env_data/torus_probs/{0}x{1}/{2}.json".format(10, 10, 1))
    get_probs = lambda n: probs[:n]
    env = EnvBuilder.sparse_grid(h, w, get_probabilities=get_probs,
                                 env_class=EnvBuilder.StochasticEnvironment, cell_len=1)
    target_node_id = env.get_vertices_number() - 1
    #env = EnvBuilder.hierarchy_test_env()
    #target_node_id = 29

    init_val = float('inf')
    best = {'mean': {}, 'median': {}}
    best_params = {'mean': {}, 'median':{}}

    for lr, fork_discount, gamma in product(lrs, fork_discounts, gammas):
        h_fsn_result = []
        q_fsn_result = []
        fsn_result = []
        print '##'*40
        print 'lr={0}, fork_discount={1}, gamma={2}'.format(lr, fork_discount, gamma)
        sys.stdout.flush()
        print 'test env #{0}'.format(env.get_name())
        target_node = env.get_vertex(target_node_id)
        for i in trange(30, leave=False): #, desc='env#{0}'.format(env.get_name()), leave=False):
            #num_steps, fsn_size = stoch_learning(target_node.coords(), env,
            #                                                    trials_number=trials_in_one_run)

            num_steps, fsn_size = hfsn.train_discrete_fsn(target_node, env,
                                                          num_episodes=trials_in_one_run,
                                                          max_episode_steps=15000, lr=lr,
                                                          fork_discount=fork_discount,
                                                          gamma=gamma, T_off=15000, verbose=0)
            #num_steps = random_trials(target_node.coords(), env, trials_number=trials_in_one_run)
            q_fsn_result.append(num_steps)

            #num_steps = periodic_Qlearning_test(target_node.coords(), env, 1000000,
            #                        0.01, alpha=0.1, gamma=0.9, penalty=0.0,
            #                        trials_number=trials_in_one_run)

            #fsn_result.append(num_steps)

            num_steps, fsn_size = hfsn.train_hierarchical_fsn(target_node, env,
                                                          num_episodes=trials_in_one_run,
                                                          max_episode_steps=15000, lr=lr,
                                                          fork_discount=fork_discount,
                                                          gamma=gamma, T_off=15000, verbose=0,
                                                          use_Q_update=False)

            h_fsn_result.append(num_steps)
            #fsn_result.append(fsn_size)

        #print 'HIERARCHICAL FSN:'
        sys.stdout.flush()
        print 'h-FSN results:'
        mdn, avr = last_10_trials_stats(h_fsn_result)
        print 'last 10 median:', mdn
        print 'last 10 average:', avr

        if best['median'].get('h_fsn', init_val) > mdn:
            best['median']['h_fsn'] = mdn
            best_params['median']['h_fsn'] = (lr, fork_discount, gamma)

        if best['mean'].get('h_fsn', init_val) > avr:
            best['mean']['h_fsn'] = avr
            best_params['mean']['h_fsn'] = (lr, fork_discount, gamma)

        sys.stdout.flush()

        print 'h-FSN with Q-update:'
        mdn, avr = last_10_trials_stats(q_fsn_result)
        print 'last 10 median:', mdn
        print 'last 10 average:', avr

        if best['median'].get('q_fsn', init_val) > mdn:
            best['median']['q_fsn'] = mdn
            best_params['median']['q_fsn'] = (lr, fork_discount, gamma)

        if best['mean'].get('q_fsn', init_val) > avr:
            best['mean']['q_fsn'] = avr
            best_params['mean']['q_fsn'] = (lr, fork_discount, gamma)


    print 'best mean:'
    print 'h_fsn:', best['mean']['h_fsn'], best_params['mean']['h_fsn']
    print 'q_fsn:', best['mean']['q_fsn'], best_params['mean']['q_fsn']
    print
    print 'best median:'
    print 'h_fsn:', best['median']['h_fsn'], best_params['median']['h_fsn']
    print 'q_fsn:', best['median']['q_fsn'], best_params['median']['q_fsn']
    print

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    mean_hdl, = ax1.plot(mean(h_fsn_result), '-b', label='mean')
    median_hdl, = ax1.plot(median(h_fsn_result), '-g', label='median')
    ax1.set_ylabel('num_steps')
    ax1.set_title('h-FSN')
    ax1.set_ylim((0,100))
    plt.legend([mean_hdl, median_hdl], ['mean', 'median'])

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(mean(q_fsn_result), '-b', label='mean')
    ax2.plot(median(q_fsn_result), '-g', label='median')
    ax2.set_xlabel('num_episodes')
    ax2.set_ylabel('num_steps')
    ax2.set_title('Discrete FSN')
    ax2.set_ylim((0, 100))

    plt.show()

    #test_periodic_envs_in_torus(5,5, env_period, 'random_envs/torus', 350)
    #test_environments(folder_with_env_files="random_envs/barabasi_albert/N25_M3",
    #                  folder_to_save_results="results/barabasi_albert/N25_M3",
    #                  env_type=altenv.ChangeStateStochasticityEnv,
    #                  trials_number=200)




    #barabasi-albert N25_M3, env №5
