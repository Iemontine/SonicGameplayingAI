import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Convolutional Neural Network
class Convolutional(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(Convolutional, self).__init__()
		# 2D convolutional layers
		self.convolution = nn.Sequential(    
			nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, stride=1, padding=1),	
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.MaxPool2d(kernel_size=2, stride=1, padding=0),	
			nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, padding=1),
			nn.ReLU(),
		)
		self.fully_connected = nn.Sequential(
			nn.Linear(32 * 6 * 6, 512),
			nn.ReLU(),
		)
		self.critic_linear = nn.Linear(512, 1)
		self.actor_linear = nn.Linear(512, out_dim)

	# 	self.initialize_weights()

	# def initialize_weights(self):
	# 	for module in self.modules():
	# 		if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
	# 			nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
	# 			# nn.init.xavier_uniform_(module.weight)
	# 			# nn.init.kaiming_uniform_(module.weight)
	# 			nn.init.constant_(module.bias, 0)

	# Forward pass
	def forward(self, x):
		x = self.convolution(x)		# Pass through convolutional layers
		x = x.view(x.size(0), -1)	# Flatten the output from the convolutional layers
		x = self.fully_connected(x)	# Pass through fully connected layer
		return self.actor_linear(x), self.critic_linear(x)