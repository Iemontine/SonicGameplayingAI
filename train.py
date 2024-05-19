import os
import torch
from ppo import PPO
import gymnasium as gym	# useful if using a wrapper for retro environments
import retro
import matplotlib.pyplot as plt

# from cnn import Convolutional
# from env import Environments
# from env import ACTION_MAPPING

# TODO: get rid of these eventually
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}, CUDA Enabled: {torch.cuda.is_available()}")

def main():
	params = {
		"gamma": 0.99,
		"lambda": 0.95,
		"learning_rate": 0.01,
		"total_timesteps": 100_000,
		"n_epochs": 10,
		"n_steps": 1024,
		"batch_size": 64,
		"clip_ratio": 1,
		"render": True,
		"epochs": 1000,
		"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
	}
	# envs = Environments(zone=1, act=1, num_envs=1)
	# model = PPO(Convolutional, num_states=envs.num_states, num_actions=len(ACTION_MAPPING), params=params)
	# model.learn(total_timesteps=100000)

	# Test stable_baselines
	# Option 1: Single environment
	env = retro.make("SonicTheHedgehog-Genesis")
	model = PPO(policy="CnnPolicy", env=env, learning_rate=params["learning_rate"], n_steps=params["n_steps"], batch_size=params["batch_size"], n_epochs=params["n_epochs"], gamma=params["gamma"], gae_lambda=params["lambda"], clip_range=params["clip_ratio"])

	# Track losses
	losses = []
	# Custom callback to track losses
	class CustomCallback(BaseCallback):
		def __init__(self, verbose=0):
			super(CustomCallback, self).__init__(verbose)
			losses = []

		def _on_step(self) -> bool:
			if 'rewards' in self.locals and self.locals['rewards'] > 0:
				print(f"Reward on {self.num_timesteps}: {self.locals['rewards']}")
			return True
	
	model.learn(total_timesteps=params["total_timesteps"], progress_bar=True, callback=CustomCallback())

	# Saving losses plot
	plt.plot(losses)
	plt.xlabel('Steps')
	plt.ylabel('Loss')
	plt.title('Loss vs Timesteps')
	plt.savefig('loss_vs_timesteps.png')

	# Option 2: Multiprocessing, after training on 200k timesteps, model did not perform well
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
	mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
	vec_env = model.get_env()
	obs = vec_env.reset()
	for i in range(1000):
		action, _states = model.predict(obs, deterministic=True)
		obs, reward, done, info = vec_env.step(action)
		vec_env.render()
	env.close()

if __name__ == "__main__":
	main()