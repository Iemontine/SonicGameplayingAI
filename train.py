import torch
import numpy as np
import matplotlib.pyplot as plt

# TODO: get rid of these eventually
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from ppo import PPO
from helpers import Callback, make_env

print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}, CUDA Enabled: {torch.cuda.is_available()}")

def main():
	# Define the parameters for the PPO model
	params = {
		# Currently arbitrarily chosen values
		"gamma": 0.75,					# Discount factor
		"tau": 0.95,					# Factor for trade-off of bias vs variance in GAE
		"beta": 0.33,					# Entropy coefficient
		"epsilon": 0.75,				# Clipped surrogate objective
		"lr": 0.003,					# Learning rate
		"steps": 1024,					# Steps before updating policy
		"batch_size": 1024,				# # Minibatch size
		"epochs": 10,					# Number of epoch when optimizing the surrogate loss
		# Development constants
		"total_timesteps": 500000,
		"policy": "CnnPolicy",
		"num_envs": 10,
		"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		"multiprocessing": True,
		"render": True,
	}

	# losses = []

	if params["multiprocessing"]:
		env = SubprocVecEnv([make_env for _ in range(params["num_envs"])])
	else:
		env = make_env()

	# Create model
	model = PPO(
		gamma=params["gamma"],
		gae_lambda=params["tau"],
		ent_coef=params["beta"],
		clip_range=params["epsilon"],
		learning_rate=params["lr"],
		n_steps=params["steps"],
		batch_size=params["batch_size"],
		n_epochs=params["epochs"],
		env=env,
		policy=params["policy"],
		device=params["device"]
	)

	# Train model
	model = PPO(params["policy"], env, verbose=1)
	model.learn(total_timesteps=params["total_timesteps"], progress_bar=True, callback=Callback())

	# plt.plot(losses)
	# plt.xlabel('Steps')
	# plt.ylabel('Loss')
	# plt.title('Loss vs Timesteps')
	# plt.savefig('loss_vs_timesteps.png')
	model.save("sonic")

	env = make_env()
	model = PPO.load("sonic", env=env)
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