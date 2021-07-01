#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from collections import deque
import torch

from vtrace import VtraceAgent
from utils import collect_trajectories, random_sample, make_mb_discounts
from parallelEnv import parallelEnv

def train(episode,env_name):
    gamma = .99
    best_score = -np.inf
    goal_score = 195.0

    nenvs = 1    #set to 1 for 
    rollout_length = 1600
    minibatches = 10*8
    # Calculate the batch_size
    nbatch = nenvs * rollout_length
    optimization_epochs = 20
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #initializing agent
    envs = parallelEnv(env_name, nenvs, seed=1234)
    #Initializing Vtrace agent
    agent = VtraceAgent(state_size=envs.observation_space.shape[0],
                     action_size=envs.action_space.n, 
                     seed=0,
                     hidden_layers=[64,64],
                     lr_policy=1e-4, 
                     use_reset=True,
                     device=device)
    print("------------------")
    print(agent.policy)
    print("------------------")

    # keep track of progress
    mean_rewards = []
    scores_window = deque(maxlen=100)
    loss_storage = []

    for i_episode in range(episode+1):
        log_probs_old, states, actions, rewards, values, dones, vals_last = collect_trajectories(envs, agent.policy, rollout_length)

        returns = np.zeros_like(rewards)
        mb_discounts = make_mb_discounts(returns.shape,gamma)     
        # convert to pytorch tensors and move to gpu if available
        returns = torch.from_numpy(returns).float().to(device).view(-1,)

        
        for _ in range(optimization_epochs):
            sampler = random_sample(nbatch, minibatches)
            for inds in sampler:
                mb_log_probs_old = log_probs_old[inds]
                mb_states = states[inds]
                mb_actions = actions[inds]
                mb_returns = returns[inds]
                loss_p, loss_v, loss_ent = agent.update(mb_log_probs_old, mb_states, mb_actions, mb_returns, mb_discounts,vals_last)
                loss_storage.append([loss_p, loss_v, loss_ent])
                
        total_rewards = np.sum(rewards, axis=0)
        scores_window.append(np.mean(total_rewards)) # last 100 scores
        mean_rewards.append(np.mean(total_rewards))  # get the average reward of the parallel environments

    
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print(total_rewards)
        if np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:            
            torch.save(agent.policy.state_dict(), "vtrace_cartpole.pth")
            best_score = np.mean(scores_window)
    
    return mean_rewards, loss_storage

#train points
mean_rewards, loss = train(400,'CartPole-v0')


#Plot parameters and scores
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 10

plt.plot(mean_rewards)
plt.ylabel('Average score')
plt.xlabel('episode')
plt.show()

#create animation
"""
from vtrace import VtraceAgent

import gym

env= gym.make('CartPole-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = PPOAgent(state_size=env.observation_space.shape[0],
                 action_size=env.action_space.n, 
                 seed=0,
                 hidden_layers=[64,64],
                 lr_policy=1e-4, 
                 use_reset=True,
                 device=device)

agent.policy.load_state_dict(torch.load('policy_cartpole.pth', map_location=lambda storage, loc: storage))

from matplotlib import animation, rc
rc('animation', html='jshtml')


# function to animate a list of frames
def animate_frames(frames):
    plt.axis('off')

    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape)==3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)  

    fanim = animation.FuncAnimation(plt.gcf(), \
        lambda x: patch.set_data(frames[x]), frames = len(frames), interval=50)
    
    return fanim

def play(env, policy, time):
    frame1 = env.reset()
    
    anim_frames = []
    
    for i in range(time):
        
        anim_frames.append(env.render(mode='rgb_array'))
        frame_input = torch.from_numpy(frame1).unsqueeze(0).float().to(device)
        action = policy.act(frame_input)['a'].cpu().numpy()
        frame1, _, is_done, _ = env.step(int(action))

        if is_done:
            print("reward :", i+1)
            break
    
    env.close()
    
    return animate_frames(anim_frames)


play(env, agent.policy, 200)

"""