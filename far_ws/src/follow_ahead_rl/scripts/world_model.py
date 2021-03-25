import torch as pt
from torch import nn, optim, distributions
from torch.nn import functional as F
import multiprocessing as mp
import numpy as np
import torch as pt
from torch import optim, distributions
from tqdm import tqdm
#from es import EvolutionStrategy
#from bipedal_walker import BipedalWalker
#from pop import Population, rollout
#from utils import ValueLogger
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from cma import CMAEvolutionStrategy

class EvolutionStrategy:
  # Wrapper for CMAEvolutionStrategy
  def __init__(self, mu, sigma, popsize, weight_decay=0.01):
    self.es = CMAEvolutionStrategy(mu.tolist(), sigma, {'popsize': popsize})
    self.weight_decay = weight_decay
    self.solutions = None

  @property
  def best(self):
    best_sol = self.es.result[0]
    best_fit = -self.es.result[1]
    return best_sol, best_fit

  def _compute_weight_decay(self, model_param_list):
    model_param_grid = np.array(model_param_list)
    return -self.weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

  def ask(self):
    self.solutions = self.es.ask()
    return self.solutions

  def tell(self, reward_table_result):
    reward_table = -np.array(reward_table_result)
    if self.weight_decay > 0:
      l2_decay = self._compute_weight_decay(self.solutions)
      reward_table += l2_decay
    self.es.tell(self.solutions, reward_table.tolist())


class WorldModel(nn.Module):
  def __init__(self, obs_dim, act_dim, hid_dim=64):
    super(WorldModel, self).__init__()
    self.obs_dim = obs_dim
    self.act_dim = act_dim 
    self.hid_dim = hid_dim

    self.lstm = nn.LSTMCell(obs_dim+act_dim, hid_dim)
    self.mu = nn.Linear(hid_dim, obs_dim)
    self.logsigma = nn.Linear(hid_dim, obs_dim)

  def forward(self, obs, act, hid):
    x = pt.cat([obs, act], dim=-1)
    h, c = self.lstm(x, hid)
    mu = self.mu(h)
    sigma = pt.exp(self.logsigma(h))
    return mu, sigma, (h, c)

class Phenotype(nn.Module):
  @property
  def genotype(self):
    params = [p.detach().view(-1) for p in self.parameters()]
    return pt.cat(params, dim=0).cpu().numpy()

  def load_genotype(self, params):
    start = 0
    for p in self.parameters():
      end = start + p.numel()
      new_p = pt.from_numpy(params[start:end])
      p.data.copy_(new_p.view(p.shape).to(p.device))
      start = end

class Controller(Phenotype, nn.Linear):
  def forward(self, obs, h):
    state = pt.cat([obs, h], dim=-1)
    return pt.tanh(super().forward(state))

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

def train_rnn(rnn, optimizer, pop, random_policy=False, 
    num_rollouts=1000, filename='ha_rnn.pt', logger=None):
  rnn = rnn.train().to(device)

  batch_size = pop.popsize
  num_batch = num_rollouts // batch_size

  batch_pbar = tqdm(range(num_batch))
  for i in batch_pbar:
    # sample rollout data
    (obs_batch, act_batch), success = pop.rollout(random_policy)
    assert success

    obs_batch = obs_batch.to(device)
    act_batch = act_batch.to(device)

    obs_batch, next_obs_batch = obs_batch[:-1], obs_batch[1:]
    hid = (pt.zeros(batch_size, rnn.hid_dim).to(device),
           pt.zeros(batch_size, rnn.hid_dim).to(device))
    rnn.zero_grad()

    # compute NLL loss
    loss = 0.0
    for obs, act, next_obs in zip(obs_batch, act_batch, next_obs_batch):
      mu, sigma, hid = rnn(obs, act, hid)
      dist = distributions.Normal(loc=mu, scale=sigma)
      nll = -dist.log_prob(next_obs) # negative log-likelihood
      nll = pt.mean(nll, dim=-1)     # mean over dimensions
      nll = pt.mean(nll, dim=0)      # mean over batch
      loss += nll
    loss = loss / len(act_batch)     # mean over trajectory
    batch_pbar.set_description(f'loss={loss.item():.3f}')

    # update RNN
    loss.backward()
    optimizer.step()

    if logger is not None:
      logger.push(loss.item())

  pt.save(rnn.state_dict(), filename)

def evolve_ctrl(ctrl, es, pop, num_gen=100, filename='ha_ctrl.pt', logger=None):
  best_sol = None
  best_fit = -np.inf

  gen_pbar = tqdm(range(num_gen))
  for g in gen_pbar:
    # upload individuals
    inds = es.ask()
    success = pop.upload_ctrl(inds)
    assert success

    # evaluate
    fits, success = pop.evaluate()
    assert success
    
    # update
    es.tell(fits)
    best_sol, best_fit = es.best
    gen_pbar.set_description(f'best={best_fit:.3f}')

    if logger is not None:
      logger.push(best_fit)

  ctrl.load_genotype(best_sol)
  pt.save(ctrl.state_dict(), filename)

def random_rollout(env, seq_len=1600):
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  obs_data = np.zeros((seq_len+1, obs_dim), dtype=np.float32)
  act_data = np.zeros((seq_len, act_dim), dtype=np.float32)
  
  obs = env.reset()
  obs_data[0] = obs
  for t in range(seq_len):
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    obs_data[t+1] = obs
    act_data[t] = act
    if done:
      obs = env.reset()

  return obs_data, act_data

def rollout(env, rnn, ctrl, seq_len=1600, render=False):
  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  obs_data = np.zeros((seq_len+1, obs_dim), dtype=np.float32)
  act_data = np.zeros((seq_len, act_dim), dtype=np.float32)
  
  obs = env.reset()
  hid = (pt.zeros(1, rnn.hid_dim), # h
         pt.zeros(1, rnn.hid_dim)) # c

  obs_data[0] = obs
  for t in range(seq_len):
    if render:
      env.render()
    obs = pt.from_numpy(obs).unsqueeze(0)
    with pt.no_grad():
      act = ctrl(obs, hid[0])
      _, _, hid = rnn(obs, act, hid)

    act = act.squeeze().numpy()
    obs, rew, done, _ = env.step(act)
    obs_data[t+1] = obs
    act_data[t] = act
    if done:
      obs = env.reset()

  return obs_data, act_data

def evaluate(env, rnn, ctrl, num_episodes=5, max_episode_steps=1600):
  fitness = 0.0
 
  for ep in range(num_episodes):
    # Initialize observation and hidden states.
    obs = env.reset()
    hid = (pt.zeros(1, rnn.hid_dim), # h
           pt.zeros(1, rnn.hid_dim)) # c

    for t in range(max_episode_steps):
      obs = pt.from_numpy(obs).unsqueeze(0)
      with pt.no_grad():
        # Take an action with the controller.
        act = ctrl(obs, hid[0])

        # Predict the next observation with the RNN.
        _, _, hid = rnn(obs, act, hid)

      # Take a step in the environment.
      act = act.squeeze().numpy()
      obs, rew, done, _ = env.step(act)

      fitness += rew
      if done:
        break

  return fitness / num_episodes

class Population:
  def __init__(self, num_workers, agents_per_worker):
    self.num_workers = num_workers
    self.agents_per_worker = agents_per_worker
    self.popsize = num_workers * agents_per_worker

    self.pipes = []
    self.procs = []
    for rank in range(num_workers):
      parent_pipe, child_pipe = mp.Pipe()
      proc = mp.Process(target=self._worker,
                        name=f'Worker-{rank}', 
                        args=(rank, child_pipe, parent_pipe))
      self.pipes.append(parent_pipe)
      self.procs.append(proc)
      proc.daemon = True
      proc.start()
      child_pipe.close()

  def _worker(self, rank, pipe, parent_pipe):
    parent_pipe.close()

    rng = np.random.RandomState(rank)

    env = BipedalWalker()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    rnn = WorldModel(obs_dim, act_dim)
    ctrls = [Controller(obs_dim+rnn.hid_dim, act_dim)
             for _ in range(self.agents_per_worker)]
  
    while True:
      command, data = pipe.recv()

      if command == 'upload_rnn': # data: rnn
        rnn.load_state_dict(data.state_dict())
        pipe.send((None, True))

      elif command == 'upload_ctrl': # data: ([inds], noisy)
        inds, noisy = data
        for ctrl, ind in zip(ctrls, inds):
          if noisy:
            ind += rng.normal(scale=1e-3, size=ind.shape)
          ctrl.load_genotype(ind)
        pipe.send((None, True))

      elif command == 'rollout': # data: random_policy
        rollouts = []
        for ctrl in ctrls:
          env.seed(rng.randint(2**31-1))
          if data: # if rollout with random policy
            trajectory = random_rollout(env)
          else:
            trajectory = rollout(env, rnn, ctrl)
          rollouts.append(trajectory)
        pipe.send((rollouts, True))

      elif command == 'evaluate': # data: None
        evaluations = []
        for ctrl in ctrls:
          env.seed(rng.randint(2**31-1))
          evaluations.append(evaluate(env, rnn, ctrl))
        pipe.send((evaluations, True))

      elif command == 'close': # data: None
        env.close()
        pipe.send((None, True))
        return True

    return False

  def upload_rnn(self, rnn):
    for p in self.pipes:
      p.send(('upload_rnn', rnn))
    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)

  def upload_ctrl(self, ctrl, noisy=False):
    if isinstance(ctrl, np.ndarray):
      for p in self.pipes:
        inds = [np.copy(ctrl) for _ in range(self.agents_per_worker)]
        p.send(('upload_ctrl', (inds, noisy)))
    elif isinstance(ctrl, list):
      start = 0
      for p in self.pipes:
        end = start + self.agents_per_worker
        inds = [np.copy(c) for c in ctrl[start:end]]
        p.send(('upload_ctrl', (inds, noisy)))
        start = end
    else:
      return False

    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)

  def rollout(self, random_policy):
    for p in self.pipes:
      p.send(('rollout', random_policy))

    rollouts = []
    all_success = True
    for rollout, success in [p.recv() for p in self.pipes]:
      rollouts.extend(rollout)
      all_success = all_success and success 

    obs_batch = []
    act_batch = []
    for obs, act in rollouts:
      obs_batch.append(obs)
      act_batch.append(act)

    # (seq_len, batch_size, dim)
    obs_batch = pt.from_numpy(np.stack(obs_batch, axis=1))
    act_batch = pt.from_numpy(np.stack(act_batch, axis=1))
    return (obs_batch, act_batch), all_success

  def evaluate(self):
    for p in self.pipes:
      p.send(('evaluate', None))

    fits = []
    all_success = True
    for fit, success in [p.recv() for p in self.pipes]:
      fits.extend(fit)
      all_success = all_success and success

    return fits, all_success

  def close(self):
    for p in self.pipes:
      p.send(('close', None))
    _, success = zip(*[p.recv() for p in self.pipes])
    return all(success)

    #

class ValueLogger:
  def __init__(self, name, bufsize=100):
    self.name = name
    self.bufsize = bufsize
    self._buffer = np.zeros((bufsize, 2))
    self._i = 0 # local iterator
    self._t = 0 # global iterator
    with open(f'{name}.csv', 'w') as f:
      f.write('step,value\n')

  def push(self, v):
    self._buffer[self._i] = (self._t, v)
    self._i += 1
    self._t += 1
    if self._i == self.bufsize:
      with open(f'{self.name}.csv', 'a') as f:
        for step, value in self._buffer:
          f.write(f'{step},{value}\n')
      self._buffer.fill(0)
      self._i = 0

  def plot(self, title, xlabel, ylabel):
    dat = pd.read_csv(f'{self.name}.csv')
    steps = dat['step']
    values = dat['value']
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(steps, values)
    plt.savefig(f'{self.name}.png')
    plt.close(fig=fig)

def main(args):
  print("IT'S DANGEROUS TO GO ALONE! TAKE THIS.")
  
  np.random.seed(0)
  pt.manual_seed(0)

  env = BipedalWalker()
  env.seed(0)

  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]

  print(f"Initializing agent (device={device})...")
  rnn = WorldModel(obs_dim, act_dim)
  ctrl = Controller(obs_dim+rnn.hid_dim, act_dim)

  # Adjust population size based on the number of available CPUs.
  num_workers = mp.cpu_count() if args.nproc is None else args.nproc
  num_workers = min(num_workers, mp.cpu_count())
  agents_per_worker = args.popsize // num_workers
  popsize = num_workers * agents_per_worker

  print(f"Initializing population with {popsize} workers...")
  pop = Population(num_workers, agents_per_worker)
  global_mu = np.zeros_like(ctrl.genotype)

  loss_logger = ValueLogger('ha_rnn_loss', bufsize=20)
  best_logger = ValueLogger('ha_ctrl_best', bufsize=100)

  # Train the RNN with random policies.
  print(f"Training M model with a random policy...")
  optimizer = optim.Adam(rnn.parameters(), lr=args.lr)
  train_rnn(rnn, optimizer, pop, random_policy=True, 
    num_rollouts=args.num_rollouts, logger=loss_logger)
  loss_logger.plot('M model training loss', 'step', 'loss')
  
  # Upload the trained RNN.
  success = pop.upload_rnn(rnn.cpu())
  assert success

  # Iteratively update controller and RNN.
  for i in range(args.niter):
    # Evolve controllers with the trained RNN.
    print(f"Iter. {i}: Evolving C model...")
    es = EvolutionStrategy(global_mu, args.sigma0, popsize)
    evolve_ctrl(ctrl, es, pop, num_gen=args.num_gen, logger=best_logger)
    best_logger.plot('C model evolution', 'gen', 'fitness')

    # Update the global best individual and upload them.
    global_mu = np.copy(ctrl.genotype)
    success = pop.upload_ctrl(global_mu, noisy=True)
    assert success
    
    # Train the RNN with the current best controller.
    print(f"Iter. {i}: Training M model...")
    train_rnn(rnn, optimizer, pop, random_policy=False,
      num_rollouts=args.num_rollouts, logger=loss_logger)
    loss_logger.plot('M model training loss', 'step', 'loss')

    # Upload the trained RNN.
    success = pop.upload_rnn(rnn.cpu())
    assert success

    # Test run!
    rollout(env, rnn, ctrl, render=True)

  success = pop.close()
  assert success

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--niter', type=int, default=10)
  parser.add_argument('--nproc', type=int, default=None)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--popsize', type=int, default=50)
  parser.add_argument('--sigma0', type=float, default=0.1)
  parser.add_argument('--num-gen', type=int, default=100)
  parser.add_argument('--num-rollouts', type=int, default=1000)
  args = parser.parse_args()

  main(args)

