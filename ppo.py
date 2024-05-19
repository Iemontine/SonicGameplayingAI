import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical
from torch.optim import Adam

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
# 	torch.nn.init.orthogonal_(layer.weight, std)
# 	torch.nn.init.constant_(layer.bias, bias_const)
# 	return layer

class PPO:
	def __init__(self, policy_type, envs, **hp):
		# note: copilot suggested gamma=0.99, lam=0.95, clip_ratio=0.2, target_kl=0.01, train_pi_iters=80, train_v_iters=80, pi_lr=3e-4, vf_lr=1e-3, train_steps=1000, epochs=50, batch_size=64, max_ep_len=1000, save_freq=10, seed=0, device='cpu'
		super().__init__()
		# note: may need to use single_action_space and single_observation_space for the action and observation space respectively
		self.actor = policy_type(np.array(envs.observation_space.shape).prod(), envs.single_action_space.n) 
		self.critic = policy_type(np.array(envs.observation_space.shape).prod(), 1) 
		self.actor_optim = Adam(self.actor.parameters(), lr=hp.learning_rate)
		self.critic_optim = Adam(self.critic.parameters(), lr=hp.learning_rate)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)
	

	# def get_value(self, x):
	# 	return self.critic(x)

	# def get_action_and_value(self, x, action=None):
	# 	logits = self.actor(x)
	# 	probs = Categorical(logits=logits)
	# 	if action is None:
	# 		action = probs.sample()
	# 	return action, probs.log_prob(action), probs.entropy(), self.critic(x)
