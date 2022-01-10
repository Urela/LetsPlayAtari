# LetsPlayAtari
This is a reposiry where I benchmark my implenation of reinforcement algorthems on Atari games. The alogorthems are implemtend in Pytorch. 

### Usage
You can train the model by executing the following command:
```bash
python atariPOO.py
```
### Results
Currently it takes roughly 45 mintues to to run 25000 time steps with 8 parallel environemnts on a 15-4300u cpu using Proximal Policy Optimization. Achieving an average run score of 5 (on past 100 games) on BreakoutNoFrameskip-v4

<p align="center">
    <img src="results/results.png" width="300"/> 
</p>
Currenlty episodes are arbitrary defined as 128 time steps. This allows me to rigidly define memory buffer size

### P.S
This is not complete work. I plan to make the algorithms faster and clearner
### Reference 
[PPO-Implementation-Deep-Dive](https://github.com/vwxyzjn/PPO-Implementation-Deep-Dive), Great starting point
[Proximal Policy Optimization - PPO in PyTorch](https://blog.varunajayasiri.com/ml/ppo_pytorch.html)
