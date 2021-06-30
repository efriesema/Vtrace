import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
import trfl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import FCNet, ConvNet, Policy

class VtraceAgent:
    def __init__(self, 
                 state_size,
                 action_size, 
                 seed,
                 hidden_layers,
                 lr_policy, 
                 use_reset, 
                 device,
                 discount=0.99,
                 entropy_cost=0.,
                 baseline_cost = 1.,         
                ):
        
        #Setting hyperparameters
        self.lr_policy = lr_policy
        self.discount = discount
        self.entropy_cost = entropy_cost
        self.baseline_cost = baseline_cost
        

        #self.main_net = ConvNet(state_size, feature_dim, seed, use_reset, input_channel).to(device)
        self.main_net = FCNet(state_size, seed, hidden_layers=[64,64], use_reset=True, act_fnc=F.relu).to(device)
        self.policy = Policy(state_size, action_size, seed, self.main_net).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.device = device
        
    

    def update(self, log_prob_old, states, actions, rewards, discounts):
    
        #traj_info = self.policy.act(states, actions)
        
        tfd = tfp.distributions
        states= states[:-1]
        actions = actions[:-1]  # [T-1]
        rewards = rewards[:-1]  # [T-1]
        discounts = discounts[:-1]  # [T-1]
        
        # Compute importance sampling weights: current policy / behavior policy.
        #behaviour_logits arethe logits from the actor network 
        behaviour_probs,policy_probs = self.policy(states)
        behaviour_probs_tf = tf.convert_to_tensor(behaviour_probs.detach().numpy())
        policy_probs_tf = tf.convert_to_tensor(policy_probs.detach().numpy())
        values = policy_probs_tf
        #logits are the logits from current policy network
        pi_behaviour = tfd.OneHotCategorical(probs=behaviour_probs_tf)
        pi_target = tfd.OneHotCategorical(probs=policy_probs_tf)
        log_rhos = pi_target.log_prob(actions) - pi_behaviour.log_prob(actions)
         

        # Critic loss.
        #vs are the vtrace targets 
        vtrace_returns = trfl.vtrace_from_importance_weights(
          log_rhos=tf.cast(log_rhos, tf.float32),
          discounts=tf.cast(self.discount * discounts, tf.float32),
          rewards=tf.cast(rewards, tf.float32),
          values=tf.cast(values[:-1], tf.float32),
          bootstrap_value=values[-1],
          )
        #values are softmax values of current policy
        critic_loss = tf.square(vtrace_returns.vs - values)

        # Policy-gradient loss.
        policy_gradient_loss = trfl.policy_gradient(
          policies=pi_target,
          actions=actions,
          action_values=vtrace_returns.pg_advantages,
           )

        # Entropy regulariser.
        entropy_loss = trfl.policy_entropy_loss(pi_target).loss


        # Combine weighted sum of actor & critic losses.
        loss = tf.reduce_mean(policy_gradient_loss +
                            self._baseline_cost * critic_loss +
                            self._entropy_cost * entropy_loss)
        
        self.optimizer.zero_grad()
        (loss).backward()
        #nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()

        return policy_gradient_loss.data.cpu().numpy(), critic_loss.data.cpu().numpy(), entropy_loss.data.cpu().numpy()