import numpy as np
import gym, time, json, os
from stochastic_torus import SparseGridToGym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import Callback

grid_result_folder = 'results/sparse_grid'
cell_size2mean_rnd = {1:210, 2:850, 4:3500, 8:14100} #only for 4x4 grids

class EpisodeLengthLogger(Callback):
  def __init__(self, verbose=False):
    super(EpisodeLengthLogger, self).__init__()
    self.verbose=verbose
    self.num_controlled_steps = {}

  def on_episode_end(self, episode, logs):
    steps = logs['nb_episode_steps']
    self.num_controlled_steps[episode] = steps
    if self.verbose:
      print 'Episode #{0}: steps: {1}'.format(episode, steps)


def run_dqn_on_env(env, nb_steps, logger=None, **kwargs):

  replay_memory_size = kwargs.pop('replay_memory_size', 5000)
  epsilon_decay_length = kwargs.pop('eps_decay_length', replay_memory_size)
  nb_steps_warmup = kwargs.pop('nb_steps_warmup', 1000)
  target_model_update = kwargs.pop('target_model_update', 1000)

  nb_actions = env.action_space.n

  # Next, we build a very simple model.
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))

  memory = SequentialMemory(limit=replay_memory_size, window_length=1)
  policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                            nb_steps=epsilon_decay_length)

  dqn = DQNAgent(model=model, nb_actions=nb_actions,
                 policy=policy, memory=memory, batch_size=32,
                 nb_steps_warmup=nb_steps_warmup, gamma=.99,
                 target_model_update=target_model_update, delta_clip=1.)

  dqn.compile(Adam(lr=1e-3), metrics=['mae'])
  callbacks = [logger] if logger else []

  dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=0,
        nb_max_episode_steps=15000, callbacks=callbacks)

# <<<<<<<<<<<<<<<<< dqn params configurations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def original_params(mean_rnd):
  params = {
    'replay_memory_size': 50*mean_rnd,
    'eps_decay_length': 50*mean_rnd,
    'nb_steps_warmup': 10*mean_rnd,
    'target_model_update':mean_rnd
  }
  return params


def burtsev_params(mean_rnd):
  params = {
    'replay_memory_size': mean_rnd,
    'eps_decay_length': mean_rnd,
    'nb_steps_warmup': mean_rnd,
    'target_model_update': mean_rnd,
  }
  return params
# <<<<<<<<<<<<<<<< /dqn params configurations >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# <<<<<<<<<<<<<<<<<<<<<<<<<<<< utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def dst_folder(grid_size, cell_size, period):
  dst = grid_result_folder + '/{0}x{0}x{1}'.format(grid_size, cell_size)
  if period is None or period <= 0:
    stoch_type = '/stochasticenvironment'
  else:
    stoch_type = '/periodicenvironment_{0}'.format(period)
  dst = dst + stoch_type
  return dst


def ensure_dir(filename):
  """Check if directories in filename exists if not creates corresponding directories!"""
  d = os.path.dirname(filename)
  if not os.path.exists(d):
    os.makedirs(d)


def save_to_json(filename, data, rewrite=True):
  """Saves data in json format"""
  mode = "w" if rewrite else "a"

  ensure_dir(filename)
  with open(filename, mode) as the_file:
    json.dump(data, the_file)


def loggers_to_list(loggers):
  all_num_steps = []
  for logger in loggers:
    run = logger.num_controlled_steps
    num_steps_in_run = [None]*len(run)
    try:
      for episode, steps in run.iteritems():
        num_steps_in_run[episode] = steps
    except IndexError as e:
      print 'episode_number:', episode, 'array size:', len(num_steps_in_run)

    all_num_steps.append(num_steps_in_run)

  return all_num_steps
# <<<<<<<<<<<<<<<<<<<<<<<<<<< /utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if __name__ == '__main__':
  import argparse
  from tqdm import trange

  parser = argparse.ArgumentParser(description='Runs DQN on specified SparseGrid envs.')
  parser.add_argument('--grid_size', type=int, default=4, choices=[4],
                      help='Size of square grid(number of cells in one side)')
  parser.add_argument('--cell_size', type=int, default=4, choices=[1,2,4,8],
                        help='Number of edges along side of each cell')
  parser.add_argument('--period', type=int, default=None,
                      help='The frequency of environmental changes.')
  parser.add_argument('--envs', type=int, nargs='+', default=range(1,11),
                      help='Indices of the envs to test on')
  parser.add_argument('--nb_steps', type=int, default=70000,
                      help='The number of steps per one DQN training.')

  args = parser.parse_args()
  print 'run with following parameters:'
  print 'grid size:', args.grid_size
  print 'cell_size:', args.cell_size
  print 'update period:', args.period
  print 'envs id:', args.envs
  print 'number of steps:', args.nb_steps

  file_template = dst_folder(args.grid_size, args.cell_size, args.period)
  file_template = file_template + '/dqn_burtsev_{0}.json'
  assert args.grid_size == 4, 'Now we test only 4x4 grids!'

  dqn_params = burtsev_params(cell_size2mean_rnd[args.cell_size])
  print 'dst files template:', file_template

  n_runs = 10
  time_per_run = []
  for env_id in args.envs:
    env = SparseGridToGym.load(args.grid_size, args.cell_size,
                               env_id, period=args.period)
    seeds = np.random.randint(0,4000, (n_runs,))
    loggers = []

    print 'Test DQN on SparseGrid({0}x{0}x{1})#{2}:'.format(
                          args.grid_size, args.cell_size, env_id
                          )
    for i in trange(n_runs):
      start_time = time.time()

      logger = EpisodeLengthLogger(verbose=False)
      loggers.append(logger)
      env.seed(seeds[i])
      run_dqn_on_env(env, args.nb_steps, logger, **dqn_params)

      duration = time.time() - start_time
      time_per_run.append(duration)

    save_to_json(file_template.format(env_id), loggers_to_list(loggers))

  print 'Total duration:', np.sum(time_per_run)/60., 'mins'
  print 'Duration per one training:', np.mean(time_per_run)/60., 'mins'
