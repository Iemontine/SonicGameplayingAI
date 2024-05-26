import gymnasium as gym	# useful if using a wrapper for retro environments
import cv2				# monitor wrapper
import retro			# stable-retro
import numpy as np
ACTION_MAPPING = {
	0: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
	1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
	2: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
	3: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
}

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env 

# Creates the environment
def make_env():
	env = retro.make('SonicTheHedgehog-Genesis')
	# Wrappers
	env = Rewarder(env)
	env = Observer(env)
	env = FrameSkipper(env, skip=8)
	env = ActionMapper(env, ACTION_MAPPING)
	return env

class Rewarder(gym.Wrapper):
	def __init__(self, env):
		super(Rewarder, self).__init__(env)
		self.env.reset()
		_, _, _, _, info = self.env.step(env.action_space.sample())
		self.start_pos = info["x"]
		self.cur_act = info["act"]
		self.progress = self.start_pos
		self.env.reset()

	def step(self, action):
		obs, _, terminated, truncated, info = self.env.step(action)
		reward = max(0, (info['x'] - self.progress))
		self.progress = max(info['x'], self.progress)
		if info["lives"] < 3:
			return obs, -1, True, False, info
		if self.cur_act != info["act"]:	# restart model on level change
			return obs, 100, True, False, info
		# TODO: multiply reward by delta x?
		return obs, reward / 100, terminated, truncated, info
	
	# potentially unused. investigating issues with resetting environment after epoch update
	# def reset(self, **kwargs):
	# 	self.env.reset()
	# 	_, _, _, _, info = self.env.step(self.env.action_space.sample())
	# 	self.start_pos = info["x"]
	# 	self.cur_act = info["act"]
	# 	self.progress = self.start_pos
	# 	return self.env.reset(**kwargs)

class Observer(gym.ObservationWrapper):
	def __init__(self, env):
		super(gym.ObservationWrapper, self).__init__(env)
		self.dim = 96
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.dim, self.dim, 1), dtype=np.uint8)

	def observation(self, obs):
		gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		resized = cv2.resize(gray, (self.dim, self.dim), interpolation=cv2.INTER_AREA)
		return resized[:, :, None]
		# return np.expand_dims(resized, axis=-1)

class FrameSkipper(gym.Wrapper):
	def __init__(self, env, skip=4):
		super(FrameSkipper, self).__init__(env)
		self.skip = skip

	def step(self, action):
		total_reward = 0.0
		done = None
		obs = None
		info = {}
		for i in range(self.skip):
			obs, reward, terminated, truncated, info = self.env.step(action)
			total_reward += reward
			if terminated or truncated:
				break
		return obs, total_reward, terminated, truncated, info

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)
	
class ActionMapper(gym.ActionWrapper):
	def __init__(self, env, action_mapping):
		super(ActionMapper, self).__init__(env)
		self.action_mapping = action_mapping
		self.action_space = gym.spaces.Discrete(len(action_mapping))

	def action(self, action):
		return self.action_mapping[action]
	
class Callback(BaseCallback):
	def __init__(self, n_steps, verbose=0):
		super(Callback, self).__init__(verbose)
		self.losses = []
		self.n_steps = n_steps

	def _on_step(self) -> bool:
		if self.verbose > 0:
			if 'rewards' in self.locals and np.any(self.locals['rewards'] > 0):
				print(f"Reward on {self.num_timesteps}: {self.locals['rewards']}")
			elif 'rewards' in self.locals and np.any(self.locals['rewards'] < 0):
				print(f"Loss on {self.num_timesteps}: {self.locals['rewards']}")

		if self.n_steps > 0 and self.num_timesteps % self.n_steps == 0:
			# self.model.env.reset()
			if self.verbose > 0:
				print(f"Policy updated on {self.num_timesteps}, saving model...")
				# TODO: save model for iterative comparisons between models
			# self.losses.append(self.locals['loss'])
		return True