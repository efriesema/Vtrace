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
        
    

    def update(self, log_probs_old, states, actions, rewards, discounts, bootstrap_values):
    
        
        tfd = tfp.distributions
        traj_info = self.policy.act(states, actions)

        # Compute importance sampling weights: current policy / behavior policy.
        #behaviour_logits arethe logits from the actor network 
        
        bootstrap_values_np= np.array(bootstrap_values)
        values = traj_info['v'].detach().numpy()
        log_rhos =traj_info['log_pi_a'] - log_probs_old
        target_probs =tf.convert_to_tensor(torch.exp(traj_info['log_pi_a']).detach().numpy())
        pi_target = tfd.Categorical(probs=target_probs)
         

        # Critic loss.
        #vs are the vtrace targets 
        print(bootstrap_values_np)
        vtrace_returns = trfl.vtrace_from_importance_weights(
          log_rhos=log_rhos,
          discounts=self.discount * discounts,
          rewards=rewards,
          values=values,
          bootstrap_value= bootstrap_values_np,
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
                            self.baseline_cost * critic_loss +
                            self.entropy_cost * entropy_loss)
        
        self.optimizer.zero_grad()
        (loss).backward()
        #nn.utils.clip_grad_norm_(self.policy.parameters(), 5)
        self.optimizer.step()

        return policy_gradient_loss.data.cpu().numpy(), critic_loss.data.cpu().numpy(), entropy_loss.data.cpu().numpy()