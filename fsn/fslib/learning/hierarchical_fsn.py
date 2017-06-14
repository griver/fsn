# coding=utf-8
from .discrete_fsn import *
from collections import deque
from itertools import takewhile
import weakref


class HierarchicalFS(FS):

    def __init__(self, parent_fs=None):
        self.set_parent(parent_fs)

    def set_parent(self, parent_fs):
        self.parent = None
        if parent_fs is not None:
            self.parent = weakref.ref(parent_fs)

    def get_parent(self):
        if self.parent is None:
            return None
        return self.parent()

    def start(self, start_state):
        pass


class SingleActionFS(HierarchicalFS):

    def __init__(self, src, dst, action, T_off, parent=None):
        super(SingleActionFS, self).__init__(parent)
        self.src = src
        self.dst = dst
        self.action = action
        self.T_off = T_off
        self.reset()


    def reset(self):
        self.last_active = -self.T_off

    def is_done(self, state):
        return True

    def choose_action(self, state, time):
        self.last_active = time
        return self.action

    def __repr__(self):
        rep = 'SAFS(src={0.src}, dst={0.dst}, T_off={0.T_off})'.format(self)
        return rep + '[la={0}]'.format(self.last_active)


class TimeExtendedFS(HierarchicalFS):

    def __init__(self, children_fs, T_off, parent=None):
        assert len(children_fs), 'TimeExtendedFS should include at least one low-level fast_fs!'
        super(TimeExtendedFS, self).__init__(parent)
        self.fs_chain = deque(children_fs)
        for fs in self.fs_chain:
            fs.set_parent(self)

        self.T_off = T_off
        self.reset()

    @property
    def src(self):
        return self.fs_chain[0].src

    @property
    def dst(self):
        return self.fs_chain[-1].dst

    def reset(self):
        self.last_active = -self.T_off
        self.inner_time = 0
        self.curr_subfs = None

    def is_done(self, state):
        assert self.inner_time <= len(self.fs_chain), \
            'inner_time(%i) > num_actions(%i)' % (self.inner_time, len(self.fs_chain))

        if self.inner_time < len(self.fs_chain):
            desired_state = self.fs_chain[self.inner_time].src
            return state != desired_state
        else:
            self.inner_time = 0
            return True

    def start(self, start_state):
        start_id = self._find_index_of_fs_with(start_state)
        assert start_id is not None,\
            '{0} started in the wrong state {1}'.format(self, start_state)
        self.inner_time = start_id
        self.curr_subfs = None

    def _find_index_of_fs_with(self, src_state, default=None):
        for i, fs in enumerate(self.fs_chain):
            if fs.src is src_state:
                return i
        return default

    def choose_action(self, state, time):
        fs = self.curr_subfs
        if fs is None or fs.is_done(state):
            fs = self.fs_chain[self.inner_time]
            fs.start(state)
            self.curr_subfs = fs
            self.inner_time += 1

        self.last_active = time
        a = fs.choose_action(state, time)
        return a

    def append_left(self, child_fs):
        self.fs_chain.appendleft(child_fs)
        child_fs.set_parent(self)

    def append_right(self, child_fs):
        self.fs_chain.append(child_fs)
        child_fs.set_parent(self)

    def remove_all_children_till(self, new_src_state):
        idx = self._find_index_of_fs_with(new_src_state)
        return [self.fs_chain.popleft() for i in xrange(idx)]


    def __repr__(self):
        path = ", ".join(str(subfs.src) for subfs in self.fs_chain)
        path += ", {0}".format(self.dst)
        repr = 'TEFS(safs=[{0}], T_off={1.T_off})'.format(path, self)
        return repr + '[la={0}, it={1}]'.format(self.last_active, self.inner_time)


class HierarchicalFSN(DiscreteFSN):

    def _add_new_fs(self, src, dst, action, init_val):

        new_fs = SingleActionFS(src, dst, action, self.T_off)

        self._src2fs.setdefault(src, []).append(new_fs)
        self._dst2fs.setdefault(dst, []).append(new_fs)

        for s in [src, dst]:
            neighbours = self._get_connected_states(s)

            if len(neighbours) >= 3:
                for tefs in self._time_extended_fs_that_crosses(s):
                    self._split_on_two(tefs, split_state=s)

            elif s is dst:
                # gets all FS that goes from dst and is not opposite to new_fs:
                branches = self.get_branches(dst, new_fs)
                if len(branches) > 0:
                    assert len(branches) == 1, 'len({}) != 1'.format(map(str, branches))
                    sucsessor_fs = branches[0]
                    parent_tefs = sucsessor_fs.get_parent()

                    if parent_tefs is not None:
                        parent_tefs.append_left(new_fs)
                    else:
                        init_val = self._fs2val[sucsessor_fs]
                        self._add_new_tefs([new_fs, sucsessor_fs], init_val)

        #init_val doesn't change if get_parent() returns None.
        init_val = self._fs2val.get(new_fs.get_parent(), init_val)
        self._fs2val[new_fs] = init_val

    def _get_connected_states(self, state):
        neighbours = set(fs.dst for fs in self._src2fs.get(state, []))
        neighbours.update(fs.src for fs in self._dst2fs.get(state, []))
        return neighbours

    def _time_extended_fs_that_crosses(self, state):
        all_fs = list(self._src2fs.get(state,[]))
        all_fs.extend(list(self._dst2fs.get(state, [])))

        all_tefs = set()
        for fs in all_fs:
            tefs = fs.get_parent()
            if tefs is None:
                continue
            if state in (tefs.src, tefs.dst):
                continue
            all_tefs.add(tefs)

        return all_tefs

    def _split_on_two(self, tefs, split_state):

        children = tefs.remove_all_children_till(split_state)
        assert children, 'spit_state equals to the tefs.src!'
        assert tefs.fs_chain, 'spit_state equals to the tefs.dst!'

        init_val = self._fs2val[tefs]

        if len(children) == 1: #if there is only one fast_fs then just update it's value
            fs = children[0]
            fs.set_parent(None)
            self._fs2val[fs] = init_val
        else:
            self._add_new_tefs(children, init_val)

        if len(tefs.fs_chain) == 1:
            fs = tefs.fs_chain[0]
            fs.set_parent(None)
            self._fs2val[fs] = init_val
            del self._fs2val[tefs]
            del tefs

    def _add_new_tefs(self, children_fs, init_val):
        new_tefs = TimeExtendedFS(children_fs, self.T_off)
        self._fs2val[new_tefs] = init_val

    def choose_fs(self, state, time):
        fs_list = list(self._src2fs.get(state, []))

        op_indices = [i for i, fs in enumerate(fs_list) if self._is_opposite(fs, self.prev_fs)]
        assert len(op_indices) <= 1, \
            'There can be only one fast_fs opposite to prev_fs!'

        for i in xrange(len(fs_list)): #if there is high level fast_fs use it:
            parent_fs = fs_list[i].get_parent()
            if parent_fs:
                fs_list[i] = parent_fs

        op_fs = fs_list[op_indices[0]] if len(op_indices) else None

        fs_list = [fs for fs in fs_list if time - fs.last_active >= fs.T_off]

        if len(fs_list) == 0:
            return self.fs_rnd

        fs_values = self._fs_values(fs_list)

        if op_fs is not None:
            for i, fs in enumerate(fs_list):
                if fs is op_fs:
                    fs_values[i] -= self.opposite_fs_penalty

        best_fs_id = np.argmax(fs_values)
        choosen_fs = fs_list[best_fs_id]
        try:
            choosen_fs.start(state)
        except AssertionError as e:
            raise e
        self.prev_fs = choosen_fs

        return choosen_fs

    def get_branches(self, dst_state, curr_fs):
        while isinstance(curr_fs, TimeExtendedFS):
            curr_fs = curr_fs.fs_chain[-1]
            assert curr_fs.dst == dst_state
        return super(HierarchicalFSN, self).get_branches(dst_state, curr_fs)

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
                if len(branches) > 1:  # if next_s is a state of choice
                    max_t_fs = max(*branches, key=lambda x: x.last_active)

                    if max_t_fs.get_parent() is None:
                        next_val = self._fs2val.get(max_t_fs)
                    else:
                        next_val = self._fs2val.get(max_t_fs.get_parent())

                    G = self.fork_discount * next_val

            G = r + self.gamma*G
            # update fast_fs value:
            old_val = self._fs2val[fs]
            val_update = self.lr * (G - old_val)
            self._fs2val[fs] = old_val + val_update


class HierarchicalQFSN(HierarchicalFSN):
    def _update_values(self, history):
        assert history[-1].next_s is self.goal, \
            '_update_network assumes that goal was achieved!'
        G = 0.
        for s, a, r, next_s, fs in reversed(history):
            if fs == self.fs_rnd:
                G = 0.
                continue

            failed = next_s is not fs.dst
            if failed:
                G = 0.  # if needed
            else:
                branches = self.get_branches(next_s, fs)
                if len(branches) > 1:  # if next_s is a state of choice

                    for i in xrange(len(branches)):  # if there is high level fast_fs use it:
                        parent_fs = branches[i].get_parent()
                        if parent_fs:
                            branches[i] = parent_fs

                    next_val = max(self._fs_values(branches))
                    G = self.fork_discount * next_val

            G = r + self.gamma * G

            # update fast_fs value:
            old_val = self._fs2val[fs]
            val_update = self.lr * (G - old_val)
            self._fs2val[fs] = old_val + val_update


def train_hierarchical_fsn(goal, env, num_episodes=250,
                           max_episode_steps=15000,
                           max_reset_attempts=100,
                           lr=0.05, fork_discount=0.99,
                           gamma=1., T_off=1500,
                           verbose=2, use_Q_update=False):

    FSNClass = HierarchicalQFSN if use_Q_update else HierarchicalFSN


    fsn = FSNClass(goal=goal, fs_rnd=RandomActionInStochEnvs(),
                           lr=lr, fork_discount=fork_discount,
                           gamma=gamma, T_off=T_off)

    stats = EpisodeStats(num_steps=[None] * num_episodes,
                         fsn_size=[None] * num_episodes)

    for episode_idx in xrange(num_episodes):

        reset_and_ensure_target_reachability(env, goal, max_reset_attempts)
        s = env.get_current_state()
        fsn.reset()
        fs, start_s, a, reward = None, None, None, 0.
        history = []
        goal_reached = False

        t_start = None

        for t in xrange(max_episode_steps):
            if fs is None or fs.is_done(state=s): #choose new fast_fs
                if fs is not None: # store transition conducted under previous fast_fs:
                    history.append(Transition(start_s, a, reward, s, fs))

                if goal_reached: # end episode if goal is reached
                    break

                fs = fsn.choose_fs(state=s, time=t)
                start_s = s
                reward = 0.

            prev_s = s
            a = fs.choose_action(s, t)
            env.update_state(a)
            s = env.get_current_state()
            goal_reached = s is goal
            reward += float(goal_reached)

            if verbose > 1:
                print t, ': {0} --> {1}, under fast_fs={2} and action={3}'.format(prev_s, s, fs, a)

        if verbose > 0:
            print 'Episode #{0} completed in {1} steps'.format(episode_idx, t)
            #if episode_idx % 10 == 0 and episode_idx:
            #    slice_l, slice_r = episode_idx-10, episode_idx
            #    print '---'*20
            #    print 'last 10 episodes mean:', np.mean(stats.num_steps[slice_l:slice_r])
            #    print 'last 10 episodes median:', np.median(stats.num_steps[slice_l:slice_r])
            #    print '---' * 20

        stats.num_steps[episode_idx] = t
        stats.fsn_size[episode_idx] = fsn.size()
        fsn.update(history, goal_reached)

    return stats
