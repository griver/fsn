import json
import sys
import os.path as path


def join_path(*arr):
  return path.abspath(path.join(*arr))

current_dir = path.dirname(path.realpath(__file__))
fslib_path = join_path(current_dir, '..', 'fsn')
torus_files_template = join_path(current_dir, '..', 'env_data/torus_probs/{n}x{n}/{env_id}.json')

#print 'fslib_path:',fslib_path
#print 'torus_files:', torus_files_template

if not fslib_path in sys.path:
  sys.path.append(fslib_path)

from fslib.env import altenv as envs
from fslib.old_fslib import EnvBuilder

def get_json_data(filename):
    """Reads data from json file"""
    data = None
    with open(filename, "r") as the_file:
        data = json.load(the_file)

    return data

def get_periodic_env_initializer(period):
  '''Returns a function that create PeriodicEnvrionment with specified period'''
  def initializer(dims, start_state_id):
    return envs.PeriodicEnvironment(period=period, dimension=dims, start_state_id=start_state_id)
  return initializer



def load_torus(grid_side, env_id, period=None):
  """
  Creates Torus environment from saved edge probabilities.

  If period agrument is not specified, then the StochasticEnvironment is used,
  otherwise the PeriodicEnvironmet with specifed period is used.
  """
  create_env = None
  if period is not None:
    create_env = get_periodic_env_initializer(period)

  probs = get_json_data(torus_files_template.format(n=grid_side, env_id=env_id))
  get_probs = lambda n: probs[:n]
  env = EnvBuilder.torus(grid_side, grid_side, env_class=create_env, get_probabilities=get_probs)
  return env

def load_sparse_grid(grid_side, cell_side, env_id, period=None):
  """
  Creates SparseGrid environment from saved edge probabilities.

  If period agrument is not specified, then the StochasticEnvironment is used,
  otherwise the PeriodicEnvironmet with specifed period is used.
  """

  if period is not None:
    if period <= 0:
      raise ValueError('period must equals to a positive integer value!')
    create_env = get_periodic_env_initializer(period)
  else:
    create_env = envs.StochasticEnvironment

  probs = get_json_data(torus_files_template.format(n=10, env_id=env_id))
  get_probs = lambda n: probs[:n]
  h = w = grid_side
  env = EnvBuilder.sparse_grid(h, w, get_probabilities=get_probs,
                        env_class=create_env, cell_len=cell_side)
  return env
