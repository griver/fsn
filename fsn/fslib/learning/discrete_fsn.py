# coding=utf-8
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ['s', 'action', 'r', 'next_s', 'fast_fs'])

EpisodeStats = namedtuple('EpisodeStats', ['num_steps', 'fsn_size'])


class FailToResetEnv(Exception):
  """
  Raised when program failed to prepare an environment
  to a new episode
  """
  pass


def reset_and_ensure_target_reachability(env, target, max_reset_attempts=100):
    for _ in xrange(max_reset_attempts):
        env.reset()
        if env.has_path_to(target):
            break
    else:
        raise FailToResetEnv("Can't find path to the target after reset.")


class FS(object):
    def reset(self):
        raise NotImplementedError('the method is not implemented yet')

    def is_done(self, state):
        raise NotImplementedError('the method is not implemented yet')

    def choose_action(self, state, time):
        raise NotImplementedError('the method is not implemented yet')


class DiscreteFS(FS):
    def __init__(self, src, dst, action, T_off):
        self.src = src
        self.dst = dst
        self.action_chain = [(src, action)]
        self.T_off = T_off
        self.reset()

    def reset(self):
        self.last_active = -self.T_off
        self.inner_time = 0

    def is_done(self, state):
        assert self.inner_time <= len(self.action_chain),\
            'inner_time(%i) > num_actions(%i)' %(self.inner_time, len(self.action_chain))

        if self.inner_time < len(self.action_chain):
            desired_state = self.action_chain[self.inner_time][0]
            return (state is not desired_state)
        else:
            self.inner_time = 0
            return True

    def choose_action(self, state, time):
        s, a = self.action_chain[self.inner_time]
        assert state is s, 'fast_fs was called on inappropriate state'
        self.inner_time += 1
        self.last_active = time
        return a

    def __repr__(self):
        repr = 'D_FS(src={0.src}, dst={0.dst}, T_off={0.T_off})'.format(self)
        return repr + '[la={0}, it={1}]'.format(self.last_active, self.inner_time)


class RandomActionInStochEnvs(FS):

    def __init__(self):
        self.dst = None
        self.src = None

    def reset(self):
        pass

    def is_done(self, state):
        return True

    def choose_action(self, state, time):
        num_actions = len(state.get_outcoming())
        return np.random.choice(num_actions)

    def __repr__(self):
        return 'RandomActionInStochEnvs()'


class DiscreteFSN(object):
    '''
    Agent that memorizes helpful transitions and
    evaluates them using temporal-difference updates
    '''
    def __init__(self, goal, fs_rnd, lr=0.05, fork_discount=0.99,
                 gamma=1., T_off=15000, init_fs_val=0.5):
        self.goal = goal
        self._src2fs = {}
        self._dst2fs = {}
        self._fs2val = {}  # Может завести еще один массив/Словарь в котором
        self.prev_fs = None  # хранить fast_fs->id, а в _values хранить id->q_val?
        self.fs_rnd = fs_rnd

        self.lr = lr
        self.fork_discount = fork_discount
        self.gamma = gamma
        self.T_off = T_off
        self.init_val = init_fs_val
        self.opposite_fs_penalty = 0.5


    def _add_new_fs(self, src, dst, action, init_val):
        new_fs = DiscreteFS(src, dst, action, self.T_off)
        self._src2fs.setdefault(src, []).append(new_fs)
        self._dst2fs.setdefault(dst, []).append(new_fs)
        self._fs2val[new_fs] = init_val

    def _fs_values(self, fs_list):
        return [self._fs2val[fs] for fs in fs_list]

    def _is_opposite(self, fs1, fs2):
        if fs1 is None or fs2 is None:
            return False
        return (fs1.src is fs2.dst) and (fs1.dst is fs2.src)

    def choose_fs(self, state, time):
        fs_list = list(self._src2fs.get(state, []))
        fs_list = [fs for fs in fs_list if time - fs.last_active >= fs.T_off]

        if len(fs_list) == 0:
            return self.fs_rnd

        fs_values = self._fs_values(fs_list)
        # decrease opposite fast_fs value:
        for i, fs in enumerate(fs_list):
            if self._is_opposite(fs, self.prev_fs):
                fs_values[i] = min(fs_values) - self.opposite_fs_penalty

        best_fs_id = np.argmax(fs_values)
        choosen_fs = fs_list[best_fs_id]
        self.prev_fs = choosen_fs

        return choosen_fs

    def reset(self):
        for fs in self._fs2val: #reset_all fast_fs
            fs.reset()

        self.prev_fs = None

    def size(self):
        #num_fs = sum(len(fs_l) for s, fs_l in self._src2fs.iteritems())
        #num_fs2 = sum(len(fs_l) for s, fs_l in self._dst2fs.iteritems())
        #assert num_fs == num_fs2, \
        #    'num_fs({0}) != num_fs2({1})'.format(num_fs, num_fs2)
        #return num_fs
        return len(self._fs2val)

    def update(self, episode_history, goal_reached):
        try:
            if goal_reached:
                # this functions assume that history ends in goal_state
                self._update_values(episode_history)
                self._update_network(episode_history)
        except AssertionError as e:
            print 'Wow!'
            raise e

    def _need_to_learn(self, transition):
        s, a, r, next_s, active_fs = transition
        if s == next_s:
            return False

        if active_fs is not self.fs_rnd:
            return False

        for fs in self._src2fs.get(s, []):
            if fs.src is s and fs.dst is next_s:
                return False

        return True

    def _update_network(self, history):
        assert history[-1].next_s is self.goal, \
            '_update_network assumes that goal was achieved!'

        for tr in reversed(history):
            if self._need_to_learn(tr):
                self._add_new_fs(src=tr.s, dst=tr.next_s,
                             action=tr.action, init_val=self.init_val)
                break
        #else:
        #    print 'Entire episode was carried out under control of the functional systems'

    def _update_values(self, history):
        assert history[-1].next_s is self.goal, \
            '_update_network assumes that goal was achieved!'

        G = 0.
        for s, a, r, next_s, fs in reversed(history):
            if fs == self.fs_rnd:
                break

            failed = next_s is not fs.dst
            if failed:
                G = 0.  # if needed
            else:
                branches = self.get_branches(next_s, fs)
                if len(branches) > 1: # if next_s is a state of choice
                    max_t_fs = max(*branches, key=lambda x: x.last_active)
                    G = self.fork_discount * self._fs2val.get(max_t_fs, 0.0)
            G = r + self.gamma * G

            # update fast_fs value:
            old_val = self._fs2val[fs]
            val_update = self.lr * (G - old_val)
            self._fs2val[fs] = old_val + val_update

    def get_branches(self, dst_state, curr_fs):
        fs_list = list(self._src2fs.get(dst_state, []))
        branches = [fs for fs in fs_list if fs.dst != curr_fs.src]
        return branches


def train_discrete_fsn(goal, env, num_episodes=250,
                       max_episode_steps=15000,
                       max_reset_attempts=100,
                       lr=0.05, fork_discount=0.99,
                       gamma=1., T_off=15000,
                       verbose=2):


    fsn = DiscreteFSN(goal=goal, fs_rnd=RandomActionInStochEnvs(),
                      lr=lr, fork_discount=fork_discount,
                      gamma=gamma, T_off=T_off)

    stats = EpisodeStats(num_steps=[None]*num_episodes,
                         fsn_size=[None]*num_episodes)

    for episode_idx in xrange(num_episodes):

        reset_and_ensure_target_reachability(env, goal, max_reset_attempts)
        s = env.get_current_state()
        fsn.reset()
        fs = None
        history = []
        goal_reached = False

        for t in xrange(max_episode_steps):
            if fs is None or fs.is_done(state=s):
                fs = fsn.choose_fs(state=s, time=t)

            a = fs.choose_action(s, t)
            env.update_state(a)
            next_s = env.get_current_state()
            goal_reached = next_s is goal
            reward = float(goal_reached)
            history.append(Transition(s, a, reward, next_s, fs))
            if verbose > 1:
                print t, ': {0} --> {1}, under fast_fs={2} and action={3}'.format(s, next_s, fs, a)

            s = next_s
            if goal_reached:
                break

        if verbose > 0:
            print 'Episode #{0} completed in {1} steps'.format(episode_idx, t)

        stats.num_steps[episode_idx] = t + 1
        stats.fsn_size[episode_idx] = fsn.size()
        fsn.update(history, goal_reached)

    return stats
