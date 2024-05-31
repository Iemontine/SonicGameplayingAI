import time
import gymnasium as gym	# useful if using a wrapper for retro environments
import cv2				# monitor wrapper
import retro			# stable-retro
import numpy as np
import os
import csv

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_checker import check_env

beat_times = []

# Create the environment
def make_env(level, pass_num, skip=0, obs_dim=(96, 96)):
	# Wrap the environment, return the function itself to allow multiprocessing
	def _init():
		env = retro.make('SonicTheHedgehog-Genesis', state=STATES[level])
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
		self.steps_to_win = 0
		self.reset_trackers()

	def step(self, action):
		obs, _, terminated, truncated, info = self.env.step(action)

		if info["lives"] < 3:						# If agent died
			self.reset_trackers()					# Reset the trackers
			if self.pass_num == 1:					# Pass 1,
				return obs, -1, True, False, info	# Reset and punish
			elif self.pass_num == 2:				# Pass 2,
				return obs, -2, True, False, info	# Reset and punish harder
		if self.cur_act != info["act"]:				# If Level change (beat level)
			self.log_win()							# Log the win
			self.steps_to_win = 0					# Reset the win trackers
			if self.pass_num == 1:					# Pass 1,
				return obs, 50, True, False, info	# Reward simply for beating the level
			elif self.pass_num == 2:				# Pass 2,
				"""Note: resetting on terminate/truncate can potentially cause negative effects on learning, as the agent may learn to reset early to avoid punishment"""
				# Reward for quick completion, may result in learning to go faster now that the solution to level has been discovered
				time_bonus = max(0, 100 - self.steps / 1000)		
				self.reset_trackers()
				return obs, time_bonus, True, False, info

		self.steps_to_win += 1
		reward = max(0, (info['x'] - self.progress))
		self.progress = max(info['x'], self.progress)
		if self.pass_num == 1:		# if agent beats level, reset and reward
			return obs, reward / 100, terminated, truncated, info
		elif self.pass_num == 2:	# if agent beats level, reset and reward, but continue to reward progress
			return obs, reward / 100 if reward > 0 else -0.01, terminated, truncated, info

	def reset_trackers(self):
		self.env.reset()
		_, _, _, _, info = self.env.step(self.env.action_space.sample())
		self.start_pos = info["x"]
		self.cur_act = info["act"]
		self.progress = self.start_pos
		self.steps = 0
		self.env.reset()

	def log_win(self):
		with open("./logs/levelbeats.csv", mode='a', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([self.steps_to_win])

	"""TODO: commented below code to investigate issues with resetting environment after epoch update (local minima being preferred, sonics stopped trying to move, although this could also be the result of a relatively high punishment (-10) for dying in combination with a relatively low reward (reward/1000) for moving forward), see below "commented out because resetting early was preventing thorough exploration of the loop problem"""
	# called when terminated/truncated
	# def reset(self, **kwargs):
	# 	self.reset_trackers()
	# 	return self.env.reset(**kwargs)
	"""it appears that resetting the trackers on both death and level change is bad. by around ~750,000 timesteps with death resets but NO victory resets, the agent learns to beat the level semi-consistently. hypotheses?"""

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
		self.n_steps = n_steps

	def format_float(self, num):
		ns = ', '.join([f'+{n:.2f}' if n >= 0 else f'{n:.2f}' for n in num])
		return ns
	
	def _on_step(self) -> bool:
		if self.verbose > 0 and 'rewards' in self.locals:
			print(f"Timestep {self.num_timesteps}:\t{self.format_float(self.locals['rewards'])}")
		if self.n_steps > 0 and self.num_timesteps % self.n_steps == 0:
			# self.model.env.reset()	# potentially prevents thorough exploration of the loop problem
			if self.verbose > 0:
				print(f"Policy updated on {self.num_timesteps}, saving model...")
				"""
				TODO: save model for iterative comparisons between models
				"""
		self.update_timesteps_in_csv()
		return True
	
	def update_timesteps_in_csv(self):
		if not os.path.exists("./logs/levelbeats.csv"):
			return
		with open("./logs/levelbeats.csv", 'r') as file:
			lines = file.readlines()
		if not lines:
			return
		last_line = [item for item in lines[-1].strip().split(',') if item != '']
		if len(last_line) == 1:
			lines[-1] = f"{last_line[0]},{self.num_timesteps}\n"
		with open("./logs/levelbeats.csv", 'w', newline='') as file:
			file.writelines(lines)

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