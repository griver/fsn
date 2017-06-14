import numpy as np
import sys
from six import StringIO

import gym
from gym import Env, spaces
from gym.utils import seeding

from torus_renderer import TorusRenderer
import utils

class BinaryGraphNodes(gym.Space):
  """
  BinaryGraphNodes represents a state space for Torus environment.
  """
  def __init__(self, num_nodes):
    self.num_nodes = num_nodes
    self.n_dims = int(np.ceil(np.log2(self.num_nodes)))
    self._int2str_pattern = '{0:0' + str(self.n_dims) + 'b}'

  def sample(self):
    node_id = np.random.randint(self.num_nodes)
    bin_array = map(int, self._int2str_pattern.format(node_id))
    return np.array(bin_array)

  def contains(self, bit_array):
    if not isinstance(bit_array, (list, np.ndarray)):
      return False
    if len(bit_array) != self.n_dims:
      return False
    try:
      bitrepr =''.join(map(str, bit_array))
      val = int(bitrepr, 2)
      return (0 <= val <= self.num_nodes)
    except ValueError as val:
      return False

  @property
  def shape(self):
    return (self.n_dims,)

  def __repr__(self):
    return 'BinaryGraphNodes(num_nodes={0})'.format(self.num_nodes)

  def __eq__(self, other):
    return (type(other) is BinaryGraphNodes) and other.num_nodes == self.num_nodes


class FailToResetEnv(Exception):
  """
  Raised when program failed to prepare an environment
  to a new episode
  """
  pass

class TorusToGym(Env):
  """Adapter that adapts torus envionment to the gym.Env interface"""
  metadata = {'render.modes': ['human', 'ansi']}
  num_actions = 4
  
  @staticmethod
  def load(grid_side, env_id, period=None):
    real_torus = utils.load_torus(grid_side, env_id, period)
    return TorusToGym(real_torus)  
  
  
  def __init__(self, torus_env, target_reward=1., step_reward=0.):
    #Set these in ALl subclasses
    n_states = torus_env.get_vertices_number()
    self.action_space = spaces.Discrete(self.num_actions)
    self.observation_space = BinaryGraphNodes(n_states)

    self._env = torus_env
    self.target = self._torus_central_state()
    self.done = True # if True self._step doesn't update the environment
    self._reset() # if successful sets self.done to False

    self.target_reward = target_reward
    self.step_reward = step_reward
    self.reward_range = (step_reward, target_reward)
    self.renderer = TorusRenderer(torus_env)


  def _step(self, action):
    """
    Accepts an action and returns a tuple (new_observation, reward, done, info).
    Args:
      - action (object): an action provided by the environment

    Returns:
      - new_observation (object): agent's observation of the current environment
      - reward (float) : amount of reward returned after previous action
      - done (boolean): whether the episode has ended, in which case further
        step() calls will return undefined results
      - info (dict): contains auxiliary diagnostic information
        (helpful for debugging, and sometimes learning)
    """

    if self.done:
      return (self._state2observation(self.target), 0.0, self.done, {})

    self._env.update_state(action)
    new_state = self._env.get_current_state()
    new_observation = self._state2observation(new_state)
    self.done = new_state is self.target
    reward = self.target_reward if self.done else self.step_reward
    return (new_observation, reward, self.done, {})


  def _render(self, mode='human', close=False):
    if close:
      return
    outfile = StringIO() if mode == 'ansi' else sys.stdout
    current_state = self._env.get_current_state()
    self.renderer.render(outfile, current_state, self.target)
    if mode != 'human':
        return outfile


  def _reset(self, max_reset_attempts=100):
    """
    Resets the state of the environment and returns an initial obseravtion
    """
    for _ in xrange(max_reset_attempts):
      self._env.reset()
      if self._env.has_path_to(self.target):
        break
    else:
      raise FailToResetEnv("Can't find path to the target after reset.")

    self.done = False
    init_state = self._env.get_current_state()
    return self._state2observation(init_state)


  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  @staticmethod
  def _state2observation(state):
    return np.array(map(int, state.coords()))


  def _torus_central_state(self):
    side_x = side_y = int(np.sqrt(self._env.get_vertices_number()))
    int_id = (side_x//2)*side_y + side_y//2
    return self._env.get_vertex(int_id)
