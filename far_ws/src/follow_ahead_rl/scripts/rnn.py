import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import Variable

import argparse
from tqdm import tqdm
from time import sleep
import sys
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid
# from torch import nn, optim, distributions
import cma
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from torch.multiprocessing import Process, Queue

from torchvision import transforms
#import torchvision.transforms as T
#from typing import Tuple

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)), #TODO: NEED TO KNOW WHAT THIS IS
    transforms.ToTensor()
])
device = 'cpu' # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False # torch.cuda.is_available()

torch.manual_seed(1)


FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
# Tensor = FloatTensor


parser = argparse.ArgumentParser()

parser.add_argument('--logdir', default='model_weights/world_model', type=str, help='Where everything is stored.')
parser.add_argument('--display', default=True, action='store_true', help="Use progress bars if specified.")
args = parser.parse_args()
time_limit = 1000
print(f'args.display {args.display}')


###############################  hyper parameters  #########################
#ENV_NAME = 'gazeboros-v0' # 'gazeborosAC-v0'  # environment name
ENV_NAME = 'LunarLander-v2'


RANDOMSEED = 42  # random seed
 
############################################################################
LR = 0.001 
reward = 1
#gaussian = 5


def gmm_loss(batch, mus, sigmas, logpi, reduce:bool = True):
    batch = batch.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(batch)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)

    log_prob = max_log_probs.squeeze() + torch.log(probs)
    if reduce:
        return -torch.mean(log_prob)
    return -log_prob

class RNN(nn.Module):
    def __init__(self, obs_dim, act_dim, gaussian, hid_dim=64, drop_prob = 0.5):
        super(RNN, self).__init__()
        self.action_dim = act_dim
        self.observation_dim = obs_dim 
        self.hidden = hid_dim
        self.gaussian_mix = gaussian
        self.reward = reward
        self.learning_r = LR 
        gmm_out = (2*obs_dim+1) * gaussian + 2

        self.rnn = nn.LSTMCell(obs_dim+act_dim, hid_dim)
        self.fc = nn.Linear(obs_dim+hid_dim, act_dim)
        self.mu = nn.Linear(hid_dim, obs_dim)
        self.logsigma = nn.Linear(hid_dim, obs_dim)
    
    def forward(self, obs, act, hid):
        x = torch.cat([act, obs], dim=-1)
        h, c = self.rnn(x, hid)
        mu = self.mu(h)
        #print("what is mu", mu)
        sigma = torch.exp(self.logsigma(h))
        #print("what is sigma", sigma)
        #print("this is the next state",c)
        #print("this is the next hidden",h)
        return mu, sigma, (h, c)
      
    def step(self, obs, h):
      state = torch.cat([obs, h], dim=-1)
      action = self.fc(state)
      return action

    

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, actions, recurrents = 64):
      super().__init__()
      self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, obs, h):
      #print("we are in controller class?")
      state = torch.cat([obs, h], dim=-1)
      return self.fc(state)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

#print("#################### RNN WEIGHTS SAVED #########################")
class RolloutGenerator(object):
    """ Utility to generate rollouts.
    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.
    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v0 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, rnn, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        #references: https://github.com/ctallec/world-models/blob/master/utils/misc.py
        ctrl_file = join(mdir, 'ctrl', 'best.tar')
        obs_dim = 8
        act_dim = 4
        self.model = rnn
        # TODO uncommet
        #self.model.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        self.controller = Controller(obs_dim , act_dim).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make(ENV_NAME)
        self.device = device

        self.time_limit = time_limit
    
    def get_action_and_transition(self, hidden, seq_len=1600, render=False):
        """ Get action and transition.
        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.
        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor
        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        obs_dim = self.env.observation_space.shape[0] #this is for the lunar lander environment
        #act_dim = env.action_space.shape[0]
        act_dim = self.env.action_space.n
        #print(obs_dim)
        #print(seq_len)
        #obs_dim_ze = np.zeros((seq_len, obs_dim))
        obs_data = np.zeros((seq_len+1, obs_dim))
        act_data = np.zeros((seq_len, act_dim))

        obs = self.env.reset()
        hid = (torch.zeros(1, 64), # h
                torch.zeros(1, 64)) # c

        obs = torch.from_numpy(obs).unsqueeze(0)
        act = self.controller.forward(obs, hid[0])
        #act = torch.tensor([[0,0 ,0 ,0]])
        _, _, hid = self.model.forward(obs, act, hid)
        m = nn.Softmax(dim=1)
        act = m(act) 
        act = torch.argmax(act)
        act = act.cpu().numpy()
        #_, latent_mu, _ = self.vae(obs)
        #action = self.controller(latent_mu, hidden[0])
        #_, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden)
        return act, obs
  
    def rollout(self, params, render=False):
        """ Execute a rollout and returns minus cumulative reward.
        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.
        :args params: parameters as a single 1D np array
        :returns: minus cumulative reward
        """
        # copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # This first render is required !
        self.env.render()

        hidden = [
            torch.zeros(1, 64).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0
        while True:
            action, hidden = self.get_action_and_transition(hidden)
            obs, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()

            cumulative += reward
            #print(cumulative)
            if done or i > self.time_limit:
                return - cumulative
            i += 1
            #break

tmp_dir = join(args.logdir, 'weights')
if not exists(tmp_dir):
    mkdir(tmp_dir)
else:
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))

################################################################################
#                           Thread routines                                    #
################################################################################
def slave_routine(p_queue, r_queue, e_queue, p_index, model):
    """ Thread routine.
    Threads interact with p_queue, the parameters queue, r_queue, the result
    queue and e_queue the end queue. They pull parameters from p_queue, execute
    the corresponding rollout, then place the result in r_queue.
    Each parameter has its own unique id. Parameters are pulled as tuples
    (s_id, params) and results are pushed as (s_id, result).  The same
    parameter can appear multiple times in p_queue, displaying the same id
    each time.
    As soon as e_queue is non empty, the thread terminate.
    When multiple gpus are involved, the assigned gpu is determined by the
    process index p_index (gpu = p_index % n_gpus).
    :args p_queue: queue containing couples (s_id, parameters) to evaluate
    :args r_queue: where to place results (s_id, results)
    :args e_queue: as soon as not empty, terminate
    :args p_index: the process index
    """
    # init routine
    #gpu = p_index % torch.cuda.device_count()
    #device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    #print("we are in slave_routine")
    # redirect streams
    #sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    #sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')
    #print(p_queue)
    with torch.no_grad():
        r_gen = RolloutGenerator(args.logdir, model, device, time_limit)

        while e_queue.empty():
            if p_queue.empty():
              #print("we are in if statement")
              sleep(.1)
            else:
              #print("we are in else statement")
              s_id, params = p_queue.get()
              #print("we are putting stuff in r_queue")
              r_queue.put((s_id, r_gen.rollout(params)))



################################################################################
#                           Controller                                         #
################################################################################

def unflatten_parameters(params, example, device):
    """ Unflatten parameters.
    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters
    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

def flatten_parameters(params):
    """ Flattening parameters.
    :args params: generator of parameters (as returned by module.parameters())
    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

def load_parameters(params, controller):
    """ Load flattened parameters into controller.
    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


if __name__ == '__main__':
  
  np.random.seed(0)
  torch.manual_seed(0)    
  env =gym.make(ENV_NAME)
  #env = gym.make('CartPole-v0').unwrapped

  #print(env)
  env.seed(0)
  obs_dim = env.observation_space.shape[0] #this is for the lunar lander environment
  #act_dim = env.action_space.shape[0]
  act_dim = env.action_space.n

  print("obs_dim + act_dim", obs_dim+act_dim)
  model = RNN(obs_dim, act_dim, 5)

  model.zero_grad()
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]

  optimizer = optim.RMSprop(model.parameters())
  criterion = torch.nn.MSELoss()

  batch_size = 100
  ls = []
  losses = []
  num_episodes = 10

  print("#################### LETS DO THE RNN #########################")
  try:
    filename= args.logdir + 'my_rnn.pt'
    #model.load_state_dict(pt.load(filename)) # TODO added by ali
    #print(f'LSTM weights loaded')
  except Exception as e:
    print(f'Error is loading weights, file might not be found or model may have changed\n{e}')

  for i_episode in range(num_episodes):
      # Initialize the environment and state
      state = env.reset()
      state = torch.from_numpy(np.array([state]))
      hid = (torch.zeros(batch_size, model.hidden).to(device),torch.zeros(batch_size, model.hidden).to(device))
      ls.append(state)
      loss = 0.0
      obs_batch, next_obs_batch = ls[:-1],ls[1:]
      for i in range(0, 10): #TODO make this more than 10
         
          ls.append(state)
          next_obs = ls[1:]
          next_obs = next_obs[0]

          act_batch = len(ls)
          #action = torch.tensor([[0,0 ,0 ,0]]) #TODO figure out how to get action state.
          pred = model.step(state, hid[0])
          m = nn.Softmax(dim=1)
          act = m(pred) 
          act = torch.argmax(act)
          act = act.cpu().numpy()
          mu, sigma, hid = model.forward(state, pred, hid)
          dist = distributions.Normal(loc=mu, scale=sigma)
          nll = -dist.log_prob(next_obs) # negative log-likelihood
          nll = torch.mean(nll, dim=-1)     # mean over dimensions
          nll = torch.mean(nll, dim=0)      # mean over batch
          loss += nll
          loss = loss / len(ls)     # mean over trajectory
          val = loss.item()

          state = env.step(act)
          state = torch.from_numpy(np.array([state[0]]))
          #print("#################### UPDATED STATE #########################")
      losses.append(val)
  torch.save(model.state_dict(), filename)



  #hard coded for testing!!
  n_samples = 2
  pop_size = 2
  num_workers = 4
  time_limit = 1000


  cur_best = None


  p_queue = Queue()
  r_queue = Queue()
  e_queue = Queue()

  print("#################### QUEUES ARE INITIALIZED #######################")


  for p_index in range(num_workers):
      Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index, model)).start()


  print("#################### PROCESSING COMPLETE #########################")


  def evaluate(solutions, results, rollouts=100):
      """ Give current controller evaluation.
      Evaluation is minus the cumulated reward averaged over rollout runs.
      :args solutions: CMA set of solutions
      :args results: corresponding results
      :args rollouts: number of rollouts
      :returns: minus averaged cumulated reward
      """
      index_min = np.argmin(results)
      best_guess = solutions[index_min]
      restimates = []

      for s_id in range(rollouts):
          p_queue.put((s_id, best_guess))

      print("Evaluating...")
      for _ in tqdm(range(rollouts)):
          while r_queue.empty():
              sleep(.1)
          restimates.append(r_queue.get()[1])

      return best_guess, np.mean(restimates), np.std(restimates)


  ctrl_dir = join(args.logdir, 'ctrl')
  if not exists(ctrl_dir):
      mkdir(ctrl_dir)

  print("#################### LET US SETUP #########################")
  controller = Controller(obs_dim , act_dim)

  print("#################### CONTROLLER SETUP #########################")
  parameters = controller.parameters()
  es = cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1,
                                {'popsize': pop_size})
  
  target_return = 100
  epoch = 0
  log_step = 3



  print("#################### ABOUT TO RUN CONTROLLER TRAINING #########################")
  while not es.stop():
      if cur_best is not None and - cur_best > target_return:
          print("Already better than target, breaking...")
          break

      r_list = [0] * pop_size  # result list. like np.zeros(pop_size).tolist()
      solutions = es.ask()

      # push parameters to queue
      for s_id, s in enumerate(solutions):
          for _ in range(n_samples):
              p_queue.put((s_id, s))

      #print("we just put something in p_queue")
      
      # retrieve results
      if args.display:
          pbar = tqdm(total=pop_size * n_samples)
      print("pbar was done!!!")
    
      for _ in range(pop_size * n_samples):
        print("We are in this for loop?")
        while r_queue.empty():
            sleep(.1)
        r_s_id, r = r_queue.get()
        r_list[r_s_id] += r / n_samples

      
        if args.display:
            pbar.update(1)

      if args.display:
          pbar.close()

      es.tell(solutions, r_list)
      es.disp()

      # evaluation and saving
      if epoch % log_step == log_step - 1:
          best_params, best, std_best = evaluate(solutions, r_list)
          print("Current evaluation: {}".format(best))
          if not cur_best or cur_best > best:
              cur_best = best
              print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
              load_parameters(best_params, controller)
              torch.save(
                  {'epoch': epoch,
                  'reward': - cur_best,
                  'state_dict': controller.state_dict()},
                  join(ctrl_dir, 'best.tar'))
          if - best > target_return:
              print("Terminating controller training with value {}...".format(best))
              break


      epoch += 1

  es.result_pretty()
  e_queue.put('EOP')



##################### ALI'S NOTES ON THE LSTM ###################################

# Select and perform an action
# need to get state data and make a sequence of the state data 
# zero pre-padding 
'''
np.array([
np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step])
,np.array(state_at_a_time_step]) #current one should be very last timestep 
])
'''
#seq_len = look back window
#       #if len(state_array)== 10:
#    then pred = model.forward(state_array, action, hid )
#else: 
#    state 
#    action = [0]
#ls = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#env = new env 

# while True:
#     state = env.step()
#     ls.append(state)

# input = ls[:-10] # 9 zeros, and new state #look at engineer man on youtube!!!! sentence generation and lyric generation

# lstm(input)


# if env is done:
#    break
