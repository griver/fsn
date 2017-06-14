from torus_adapter import TorusToGym, BinaryGraphNodes, spaces
import utils


class SparseGridToGym(TorusToGym):

  @staticmethod
  def load(grid_side, cell_side, env_id, period=None):
    grid = utils.load_sparse_grid(grid_side, cell_side,
                                  env_id, period=period)
    return SparseGridToGym(grid)


  def __init__(self, sparse_grid_env, target_reward=1., step_reward=0.):
    n_states = sparse_grid_env.get_vertices_number()
    self.action_space = spaces.Discrete(self.num_actions)
    self.observation_space = BinaryGraphNodes(n_states)

    self._env = sparse_grid_env
    self.target = self._env.get_vertex(-1)
    self.done = True # if True self._step doesn't update the environment
    self._reset() # if successful sets self.done to False

    self.target_reward = target_reward
    self.step_reward = step_reward
    self.reward_range = (step_reward, target_reward)


  def _render(self, mode='human', close=False):
    if close:
      return
    return None
