import gym
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def make_env(gym_id, idx, seed, record=False, run_name=''):
  def thunk():
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record and idx==0:
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    # seeds for reproductivity
    env.seed(seed)                   
    env.action_space.seed(seed)      
    env.observation_space.seed(seed) 
    return env
  return thunk

class ActorCritic(nn.Module):
  def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

  def __init__(self, in_space, out_space, lr=2.5e-4):
    super(ActorCritic, self).__init__()

    self.conv = nn.Sequential(
      self.layer_init(nn.Conv2d(4,  32, 8, stride=4)), nn.ReLU(),
      self.layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
      self.layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
      nn.Flatten()
    )

    self.fc_critic = nn.Sequential(
      self.layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
      self.layer_init(nn.Linear(512, 1), std=1)
    )

    self.fc_actor = nn.Sequential(
      self.layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
      self.layer_init(nn.Linear(512, out_space.n), std=0.01),
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
    self.to('cpu')

  def actor(self, state):
    state = self.conv(state)
    dist = self.fc_actor(state)
    dist = Categorical(logits=dist)
    return dist

  def critic(self, state):
    state = self.conv(state)
    value = self.fc_critic(state)
    return value

class PPO:  
  def __init__(self, in_space, out_space, batch_size=4, Mem_size=128, num_envs=4):
    self.lr    = 2.5e-4   # 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2

    self.AC = ActorCritic(in_space, out_space, self.lr)

    self.states  = np.zeros((Mem_size, num_envs)+in_space.shape,  dtype=np.float32)
    self.actions = np.zeros((Mem_size, num_envs)+out_space.shape, dtype=np.int32)
    self.rewards = np.zeros((Mem_size, num_envs), dtype=np.float32)
    self.probs   = np.zeros((Mem_size, num_envs), dtype=np.float32)
    self.values  = np.zeros((Mem_size, num_envs), dtype=np.float32)
    self.dones   = np.zeros((Mem_size, num_envs), dtype=np.int32)
    self.size, self.bsize, self.idx = Mem_size, batch_size, 0

  def selectAction(self, obs):
    obs    = torch.from_numpy(obs).float().to('cpu')
    value  = self.AC.critic(obs)
    dist   = self.AC.actor(obs)
    action = dist.sample()

    probs  = torch.squeeze(dist.log_prob(action)).detach().numpy()
    action = torch.squeeze(action).detach().numpy()
    value  = torch.squeeze(value).detach().numpy()
    return action, probs, value

  def store(self, state, action, reward, probs, vals, done):
    idx = self.idx % self.size
    self.idx += 1
    self.states  [idx] = state
    self.actions [idx] = action
    self.rewards [idx] = reward
    self.probs   [idx] = probs
    self.values  [idx] = vals
    self.dones   [idx] = 1-(done)

  def train(self):
    for epi in range( self.epoch):
      # finding Advantages using gamma returns
      nvalues = np.concatenate([self.values[1:] ,[self.values[-1]]])
      delta = self.rewards + self.gamma*nvalues* self.dones - self.values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu').reshape(-1)
      values    = torch.tensor(self.values.reshape(-1)).to('cpu')
      
      # create mini batches
      indices = np.arange( self.size, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in range(0, self.size, self.bsize)]

      # reward Annealing
      frac = 1.0 - (epi - 1.0) / self.epoch
      lrnow = frac * self.lr
      self.AC.optimizer.param_groups[0]["lr"] = lrnow

      b, n, c, h, w = self.states.shape
      _states = self.states.reshape(b*n ,c, h, w)
      _actions = self.actions.reshape(-1)
      _probs = self.probs.reshape(-1)

      for batch in batches:
        states    = torch.tensor(_states[batch], dtype=torch.float).to('cpu')
        actions   = torch.tensor(_actions[batch], dtype=torch.float).to('cpu')
        old_probs = torch.tensor(_probs[batch], dtype=torch.float).to('cpu')

        #print( len(batch) )
        #print( states.shape)
        #print( states.shape)


        dist = self.AC.actor(states)
        crit = self.AC.critic(states)
        crit = torch.squeeze(crit)

        # Finding the ratio (pi_theta / pi_theta__old)
        #new_probs = dist.log_prob(actions)
        #print(actions.shape)
        new_probs  = torch.squeeze(dist.log_prob(actions))

        #print( new_probs.shape, old_probs.shape)
        ratio = new_probs.exp() / old_probs.exp()

        # Finding Surrogate Loss
        #print( new_probs.shape, old_probs.shape)
        #print( ratio.shape, advantage[batch].shape)
        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        returns = advantage[batch] + values[batch]

        # final loss of clipped objective PPO: loss = actor_loss + 0.5*critic_loss 
        loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-crit)**2).mean()

        # take gradient step
        self.AC.optimizer.zero_grad()
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.AC.parameters(), 0.5)
        self.AC.optimizer.step()

    self.idx=0
    pass

# Global variables
gym_id = "BreakoutNoFrameskip-v4"
seed        = 1
num_envs    = 8       # number of parallel environments

# Environment Hyperparameters
max_ep_len = 128                    # max timesteps in one episode
max_training_timesteps = 25000 #int(1e5)   # break training loop if timeteps > max_training_timesteps
update_timestep = max_ep_len * 1    # update policy every n timesteps
print_freq = update_timestep        # print avg reward in the interval (in num timesteps)

print_running_reward   = 0
print_running_episodes = 0


envs = gym.vector.SyncVectorEnv([
      make_env(gym_id, i, seed+i, record=False, run_name=f"{gym_id}__{int(time.time())}" ) 
          for i in range(num_envs)
      ])

time_step  = 0 
i_episode = 0


agent = PPO(envs.single_observation_space, envs.single_action_space, batch_size=4, Mem_size=update_timestep, num_envs=num_envs)

scores, avg_scores, time_step_arr = [], [], []

while time_step <= max_training_timesteps:
  state = envs.reset()
  for t in range(1, max_ep_len+1):

    state = state/255
    action, probs, val = agent.selectAction(state)
    _state, reward, done, info = envs.step(action)
    agent.store(state, action, reward, probs, val, done)
    state = _state

    time_step += 1

    # train PPO agent
    if time_step % update_timestep == 0: agent.train()

    for item in info:
      if "episode" in item.keys():
        score = item['episode']['r']
        scores.append( score  )
        avg = np.round(np.mean(scores[-100:]), 2)
        avg_scores.append(avg)
        time_step_arr.append(time_step)
        i_episode+=1
        print(f"Episode: {i_episode}  Episodic return: {score} Avg returns:{avg} Time step: {time_step} ")

from bokeh.plotting import figure, show
p = figure(title="Atari [BreakoutNoFrameskip-v4]", x_axis_label="Time steps", y_axis_label="Scores")
#x = np.arange(len(avg_scores))
x = time_step_arr
p.line(x, avg_scores,  legend_label="scores", line_color="blue", line_width=2)
show(p) 
