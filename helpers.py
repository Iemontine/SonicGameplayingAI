import time
import gymnasium as gym	# useful if using a wrapper for retro environments
import cv2				# monitor wrapper
import retro			# stable-retro
import numpy as np
import os

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_checker import check_env

# Creates the environment
def make_env(level, pass_num, skip=0, obs_dim=(96, 96)):
	def _init():
		env = retro.make('SonicTheHedgehog-Genesis', state=STATES[level])
		# Wrappers
		env = Rewarder(env, pass_num=pass_num)
		env = Observer(env, dim=obs_dim)
		env = FrameSkipper(env, skip=skip)
		env = ActionMapper(env, ACTION_MAPPING)
		return env
	return _init

class Rewarder(gym.Wrapper):
	def __init__(self, env, pass_num):
		super(Rewarder, self).__init__(env)
		self.pass_num = pass_num
		self.steps = 0
		self.reset_trackers()

	def step(self, action):
		if self.pass_num == 1:
			obs, _, terminated, truncated, info = self.env.step(action)
			reward = max(0, (info['x'] - self.progress))
			self.progress = max(info['x'], self.progress)
			# if agent died, reset and punish
			if info["lives"] < 3:
				self.reset_trackers()
				return obs, -1, True, False, info
			# if agent beats level, reset and reward
			if self.cur_act != info["act"]:
				return obs, 50, True, False, info
			# TODO: multiply reward by delta x?
			return obs, reward / 100, terminated, truncated, info
		elif self.pass_num == 2:
			self.steps += 1
			obs, _, terminated, truncated, info = self.env.step(action)
			reward = max(0, (info['x'] - self.progress))
			self.progress = max(info['x'], self.progress)
			# if agent died, reset and punish harder
			if info["lives"] < 3:
				self.reset_trackers()
				return obs, -1, True, False, info
			# if agent beats level, reset and reward, but continue to reward progress
			if self.cur_act != info["act"]:
				"""
				investigating what happens when progress is reset when level changes
				could result in learning to go FASTER once the solution to a level is found
				now that we know that resetting trackers on level change AND death during initial training negatively effects ability to learn to win, perhaps optimization for speed can occur
				after initial training, once the agent has at least learned to win
				"""
				time_bonus = max(0, 1000 - self.steps) / 100.0
				self.reset_trackers()
				return obs, 50 + time_bonus, True, False, info
			return obs, reward / 100 if reward > 0 else -0.01, terminated, truncated, info

	def reset_trackers(self):
		self.env.reset()
		_, _, _, _, info = self.env.step(self.env.action_space.sample())
		self.start_pos = info["x"]
		self.cur_act = info["act"]
		self.progress = self.start_pos
		self.steps = 0
		self.env.reset()
	"""
	TODO: commented below code to investigate issues with resetting environment after epoch update (local minima being preferred, sonics stopped trying to move, although this could also be the result of a relatively high punishment (-10) for dying in combination with a relatively low reward (reward/1000) for moving forward), see below "commented out because resetting early was preventing thorough exploration of the loop problem"
	"""
	# called when terminated/truncated
	# def reset(self, **kwargs):
	# 	self.env.reset()
	# 	_, _, _, _, info = self.env.step(self.env.action_space.sample())
	# 	self.start_pos = info["x"]
	# 	self.cur_act = info["act"]
	# 	self.progress = self.start_pos
	# 	return self.env.reset(**kwargs)
	"""
	more notes:
	it appears that resetting the trackers on both death and level change is bad. by around ~750,000 timesteps with death resets but NO victory resets, the agent learns to beat the level semi-consistently.
	hypotheses?
	"""

class Observer(gym.ObservationWrapper):
	def __init__(self, env, dim=(96, 96)):
		super(gym.ObservationWrapper, self).__init__(env)
		self.dimX = dim[0]
		self.dimY = dim[1]
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.dimX, self.dimY, 1), dtype=np.uint8)

	def observation(self, obs):
		gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
		resized = cv2.resize(gray, (self.dimX, self.dimY), interpolation=cv2.INTER_AREA)
		return resized[:, :, None]
		# return np.expand_dims(resized, axis=-1)

class FrameSkipper(gym.Wrapper):
	def __init__(self, env, skip=4):
		super(FrameSkipper, self).__init__(env)
		self.skip = skip

	def step(self, action):
		total_reward = 0.0
		terminated = None
		truncated = None
		obs = None
		info = {}
		for _ in range(self.skip):
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
	# 	self.log_dir = "logs/"
	# 	self.save_path = os.path.join("logs/", "best_model")
	# 	self.best_mean_reward = -np.inf

	# def _init_callback(self) -> None:
	# 	# Create folder if needed
	# 	if self.save_path is not None:
	# 		os.makedirs(self.save_path, exist_ok=True)

	def _on_step(self) -> bool:
		if self.verbose > 0:
			if 'rewards' in self.locals and np.any(self.locals['rewards'] > 0):
				print(f"Reward on {self.num_timesteps}: {self.locals['rewards']}")
			elif 'rewards' in self.locals and np.any(self.locals['rewards'] < 0):
				print(f"Loss on {self.num_timesteps}: {self.locals['rewards']}")

		if self.n_steps > 0 and self.num_timesteps % self.n_steps == 0:
			"""below line was commented out because resetting early was preventing thorough exploration of the loop problem"""
			# self.model.env.reset()
			if self.verbose > 0:
				print(f"Policy updated on {self.num_timesteps}, saving model...")
				"""
				TODO: save model for iterative comparisons between models
				"""
			# x, y = ts2xy(load_results(self.log_dir), "timesteps")
			# if len(x) > 0:
			# 	# Mean training reward over the last 100 episodes
			# 	mean_reward = np.mean(y[-100:])
			# 	if self.verbose >= 1:
			# 		print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

			# 	# New best model, you could save the agent here
			# 	if mean_reward > self.best_mean_reward:
			# 		self.best_mean_reward = mean_reward
			# 		# Example for saving best model
			# 		if self.verbose >= 1:
			# 			print(f"Saving new best model to {self.save_path}")
			# 		self.model.save(self.save_path)
			# self.losses.append(self.locals['loss'])
		return True
	
# TODO: other implementations show that action space may need to increased for the more complex levels (after 1-2)
ACTION_MAPPING = {
	0: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],	# Left
	1: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],	# Right
	2: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],	# Right, Down
	3: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],	# Right, Jump
	# 4: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],	# Left, Down
	# 5: [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],	# Left, Jump
}
STATES = {
	"1-1": "GreenHillZone.Act1",
	"1-2": "GreenHillZone.Act2",
	"1-3": "GreenHillZone.Act3",
	"2-1": "MarbleZone.Act1",
	"2-2": "MarbleZone.Act2",
	"2-3": "MarbleZone.Act3",
	"3-1": "SpringYardZone.Act1",
	"3-2": "SpringYardZone.Act2",
	"3-3": "SpringYardZone.Act3",
	"4-1": "LabyrinthZone.Act1",
	"4-2": "LabyrinthZone.Act2",
	"4-3": "LabyrinthZone.Act3",
	"5-1": "StarLightZone.Act1",
	"5-2": "StarLightZone.Act2",
	"5-3": "StarLightZone.Act3",
	"6-1": "ScrapBrainZone.Act1",
	"6-2": "ScrapBrainZone.Act2",
	"6-3": "ScrapBrainZone.Act3",
}