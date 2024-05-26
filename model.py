from helpers import Callback, make_env
from ppo import PPO

def main():
	env = make_env()
	model = PPO.load("sonic", env=env)
	vec_env = model.get_env()
	obs = vec_env.reset()
	done = False

	while not done:
		action, states = model.predict(obs)
		obs, reward, done, info = vec_env.step(action)
		vec_env.render()
	env.close()

if __name__ == "__main__":
	main()