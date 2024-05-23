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
	gamma = trial.suggest_float('gamma', 0.9, 0.999)						# Discount factor
	tau = trial.suggest_float('tau', 0.9, 0.999)							# Factor for trade-off of bias vs variance in GAE
	beta = trial.suggest_float('beta', 0.001, 0.1)							# Entropy coefficient
	lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)							# Learning rate
	n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024])		# Steps before updating policy
	batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
	epochs = trial.suggest_int('epochs', 3, 10)
	clip_ratio = trial.suggest_float('clip_ratio', 0.1, 0.3)
	# Create vectorized environment
	params = {
		"gamma": gamma,
		"tau": tau,
		"beta": beta,
		"lr": lr,
		"epochs": epochs,
		"total_timesteps": 1000000,
		"steps": n_steps,
		"batch_size": batch_size,
		"clip_ratio": clip_ratio,
		# basically constant
		"policy": "CnnPolicy",
		"render": True,
		"multiprocessing": True,
		"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
	}
	num_envs = 10
	if params["multiprocessing"]:
		env = SubprocVecEnv([make_env for _ in range(num_envs)])
	else:
		env = make_env()
	
	# Create model
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
		device=params["device"]
	)
	model.learn(total_timesteps=params["total_timesteps"], callback=Callback())
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=True)
	return np.mean(mean_reward)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
	print(f"{key}: {value}")

# def main():
# 	params = {
# 		"gamma": 0.75,				
# 		"tau": 0.95,				
# 		"beta": 0.33,				
# 		"epsilon": 0.45,			
# 		"lr": 0.003,				
# 		"epochs": 10,
# 		"total_timesteps": 100_000_0,
# 		"steps": 512,				
# 		"batch_size": 1024,
# 		"clip_ratio": 0.75,
# 		"policy": "CnnPolicy",
# 		"render": True,
# 	}
# 	if not os.path.exists("sonic.zip"):
# 		losses = []

# 		# model.learn(total_timesteps=params["total_timesteps"], progress_bar=True, callback=Callback())
# 		# Create vectorized environments
# 		num_envs = 10
# 		env = SubprocVecEnv([make_env() for _ in range(num_envs)])
# 		model = PPO(
# 			gamma=params["gamma"], 
# 			gae_lambda=params["tau"], 
# 			ent_coef=params["beta"], 
# 			learning_rate=params["lr"], 
# 			n_steps=params["steps"], 
# 			batch_size=params["batch_size"], 
# 			n_epochs=params["epochs"], 
# 			clip_range=params["clip_ratio"],
# 			env=env, 
# 			policy=params["policy"], 
# 			device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		)
# 		model = PPO(params["policy"], env, verbose=1)
# 		model.learn(total_timesteps=params["total_timesteps"], progress_bar=True, callback=Callback())

# 		plt.plot(losses)
# 		plt.xlabel('Steps')
# 		plt.ylabel('Loss')
# 		plt.title('Loss vs Timesteps')
# 		plt.savefig('loss_vs_timesteps.png')
# 		model.save("sonic")

# 	env = make_env()
# 	model = PPO.load("sonic", env=env)
# 	vec_env = model.get_env()
# 	obs = vec_env.reset()
# 	done = False
# 	while not done:
# 		action, states = model.predict(obs, deterministic=True)
# 		obs, reward, done, info = vec_env.step(action)
# 		vec_env.render()
# 	env.close()

# if __name__ == "__main__":
# 	main()