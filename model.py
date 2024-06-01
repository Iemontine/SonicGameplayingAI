from helpers import Callback, make_env
from ppo import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from train import params

def main():
	env = SubprocVecEnv([make_env(params["level"], skip=params["frame_skip"], obs_dim=params["observation_dimension"], pass_num=2) for _ in range(5)])
	env = VecFrameStack(env, n_stack=params["frame_stack"])
	model = PPO.load("sonic2", env=env)
	vec_env = model.get_env()
	obs = vec_env.reset()
	done = False

	while not done:
		action, states = model.predict(obs)
		obs, reward, _, info = vec_env.step(action)
		done = info[0]['act'] == 1
		vec_env.render()
	env.close()

if __name__ == "__main__":
	main()