import os
import torch
from ppo import PPO
import gymnasium as gym	# useful if using a wrapper for retro environments
import retro
import matplotlib.pyplot as plt
from collections import deque

# from cnn import Convolutional
# from env import Environments
# from env import ACTION_MAPPING

# TODO: get rid of these eventually
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}, CUDA Enabled: {torch.cuda.is_available()}")

# A successful reward function for MountainCar	
class RewardWrapper(gym.Wrapper):
	def __init__(self, env):
		super(RewardWrapper, self).__init__(env)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		
		position, velocity = obs
		if velocity < 0:
			velocity /= 2
		velocity *= 100
		reward = 0.5*(velocity**2) + 0.33*abs(position - (-0.5))
		
		return obs, reward, terminated, truncated, info

class FrameSkipper(gym.Wrapper):
	def __init__(self, env, skip=4):
		super(FrameSkipper, self).__init__(env)
		self.skip = skip

	def step(self, action):
		reward = None
		done = None
		obs = None
		info = {}
		for i in range(self.skip):  # Add loop index
			obs, reward, terminated, truncated, info = self.env.step(action)
			reward = reward
			if done:
				break
		return obs, reward, terminated, truncated, info

	def reset(self, **kwargs):  # Accept additional keyword arguments
		return self.env.reset(**kwargs)
	
def main():
	# MountainCar
	params = {
		"gamma": 0.99,				# Discount factor
		"tau": 0.95,				# Factor for trade-off of bias vs variance in GAE
		"beta": 0.1,				# Entropy coefficient
		"epsilon": 0.2,				# Clipped surrogate objective
		"lr": 0.001,				# Learning rate
		"epochs": 10,
		"total_timesteps": 100_000,
		"steps": 64,				# Steps before updating policy
		"batch_size": 64,
		"clip_ratio": 0.2,
		"policy": "MlpPolicy",
		"render": True,
	}
    # model = PPO(Convolutional, num_states=envs.num_states, num_actions=len(ACTION_MAPPING), params=params)
	# model.learn(total_timesteps=100000)
	env = gym.make('MountainCar-v0', render_mode='human')
	# env = gym.make('MountainCar-v0', render_mode='human')
	# check_env(env, warn=True, skip_render_check=True)
	env = RewardWrapper(env)
	env = FrameSkipper(env, skip=1)
	if not os.path.exists("sonic.zip"):
		# Test stable_baselines x gymnasium
		# Option 1: Single environment
		# env = retro.make("SonicTheHedgehog-Genesis")
		model = PPO(
			gamma=params["gamma"], 
			gae_lambda=params["tau"], 
			ent_coef=params["beta"], 
			learning_rate=params["lr"], 
			n_steps=params["steps"], 
			batch_size=params["batch_size"], 
			n_epochs=params["epochs"], 
			clip_range=params["clip_ratio"],
			env=env, 
			policy=params["policy"], 
			device= torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		# Track losses
		losses = []
		# Called whenever a step is made
		class Callback(BaseCallback):
			def __init__(self, verbose=0):
				super(Callback, self).__init__(verbose)
				losses = []

			def _on_step(self) -> bool:
				if 'rewards' in self.locals and self.locals['rewards'] > 0:
					print(f"Reward on {self.num_timesteps}: {self.locals['rewards']}")
				elif 'rewards' in self.locals and self.locals['rewards'] < 0:
					print(f"Loss on {self.num_timesteps}: {self.locals['rewards']}")
				return True
		model.learn(total_timesteps=params["total_timesteps"], progress_bar=True, callback=Callback())
		# Saving losses plot, not working for some reason
		plt.plot(losses)
		plt.xlabel('Steps')
		plt.ylabel('Loss')
		plt.title('Loss vs Timesteps')
		plt.savefig('loss_vs_timesteps.png')
		# Option 2: Multiprocessing
		# def make_env():
		# 	def _init():
		# 		env = retro.make(game='SonicTheHedgehog-Genesis')
		# 		return env
		# 	return _init
		# # Create vectorized environments
		# num_envs = 4
		# env = SubprocVecEnv([make_env() for _ in range(num_envs)])
		# model = PPO("MlpPolicy", env, verbose=1)
		# model.learn(total_timesteps=2e5, progress_bar=True)
		model.save("sonic")
	model = PPO.load("sonic", env=env)
	# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
	vec_env = model.get_env()
	obs = vec_env.reset()
	done = False
	while not done:
		action, states = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		vec_env.render()
	env.close()
if __name__ == "__main__":
	main()