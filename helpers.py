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
from stable_baselines3.common.monitor import Monitor

# Creates the environment
def make_env():
	env = retro.make('SonicTheHedgehog-Genesis')
	env = Monitor(env)
	env = RewardWrapper(env)
	env = FramePreprocessingWrapper(env)
	env = FrameSkipper(env, skip=4)
	env = ActionMappingWrapper(env, ACTION_MAPPING)
	return env

class ActionMappingWrapper(gym.ActionWrapper):
	def __init__(self, env, action_mapping):
		super(ActionMappingWrapper, self).__init__(env)
		self.action_mapping = action_mapping
		self.action_space = gym.spaces.Discrete(len(action_mapping))

	def action(self, action):
		return self.action_mapping[action]

class RewardWrapper(gym.Wrapper):
	def __init__(self, env):
		super(RewardWrapper, self).__init__(env)
		self.env.reset()
		_, _, _, _, info = self.env.step(env.action_space.sample())
		self.start_pos = info["x"]
		self.start_score = info["score"]
		self.start_rings = info["rings"]
		self.start_lives = info["lives"]
		self.env.reset()
		self.progress = self.start_pos
		self.curr_score = self.start_score
		self.curr_rings = self.start_rings
		self.curr_lives = self.start_lives

	def step(self, action):
		obs, _, terminated, truncated, info = self.env.step(action)
		reward = min(max((info['x'] - self.progress), 0), 2)
		self.progress = max(info['x'], self.progress)
		# reward += (info["score"] - self.curr_score) / 40
		# self.curr_score = info["score"]
		if info["lives"] < 3:
			return obs, -5, False, True, info
		return obs, reward / 10, terminated, truncated, info

class FramePreprocessingWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super(FramePreprocessingWrapper, self).__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

	def observation(self, obs):
		gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
		return np.expand_dims(resized, axis=-1)

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
	
class Callback(BaseCallback):
	def __init__(self, verbose=0):
		super(Callback, self).__init__(verbose)
		self.losses = []

	def _on_step(self) -> bool:
		if 'rewards' in self.locals and np.any(self.locals['rewards'] > 0):
			print(f"Reward on {self.num_timesteps}: {self.locals['rewards']}")
		elif 'rewards' in self.locals and np.any(self.locals['rewards'] < 0):
			print(f"Loss on {self.num_timesteps}: {self.locals['rewards']}")
		return True