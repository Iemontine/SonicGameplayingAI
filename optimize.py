import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna

# TODO: get rid of these eventually
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from ppo import PPO
from helpers import Callback, make_env

print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}, CUDA Enabled: {torch.cuda.is_available()}")

def objective(trial):
	# Define the hyperparameters for Bayesian optimization via optuna
	gamma = trial.suggest_float('gamma', 0.1, 0.999)						
	tau = trial.suggest_float('tau', 0.1, 0.999)							
	beta = trial.suggest_float('beta', 0.0001, 0.1)
	epsilon = trial.suggest_float('epsilon', 0.1, 1.0)				
	lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)							
	n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048, 4096])		
	batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048, 4096])
	epochs = trial.suggest_int('epochs', 3, 10)

	# Define the parameters for the PPO model
	params = {
		"gamma": gamma,					# Discount factor
		"tau": tau,						# Factor for trade-off of bias vs variance in GAE
		"beta": beta,					# Entropy coefficient
		"epsilon": epsilon,				# Clipped surrogate objective
		"lr": lr,						# Learning rate
		"steps": n_steps,				# Steps before updating policy
		"batch_size": batch_size,		# Minibatch size
		"epochs": epochs,				# Number of epoch when optimizing the surrogate loss
		# Development constants
		"total_timesteps": 1000,
		"policy": "CnnPolicy",
		"num_envs": 10,
		"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
		"multiprocessing": False,
		"render": True,
	}

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
	model.learn(total_timesteps=params["total_timesteps"], progress_bar=True)
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, deterministic=False)
	print("Mean reward:", mean_reward)
	return np.mean(mean_reward)

def main():
	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=1)

	print("Best trial:")
	trial = study.best_trial
	print(f"Value: {trial.value}")
	print("Params: ")
	for key, value in trial.params.items():
		print(f"{key}: {value}")

if __name__ == "__main__":
	main()