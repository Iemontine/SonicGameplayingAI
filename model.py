from helpers import Callback, make_env
from ppo import PPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from train import params

NUM_ENVS = 3

def main():
    env = SubprocVecEnv([make_env(params["level"], skip=params["frame_skip"], obs_dim=params["observation_dimension"], pass_num=params["pass_num"]) for _ in range(NUM_ENVS)])
    env = VecFrameStack(env, n_stack=params["frame_stack"])
    model = PPO.load("sonic3", env=env)
    vec_env = model.get_env()
    obs = vec_env.reset()
    all_done = [False] * NUM_ENVS
    while all_done != [True, True, True]:
        action, _ = model.predict(obs)
        obs, reward, done, info = vec_env.step(action)
        for i in range(NUM_ENVS):
            if done[i]:
                all_done[i] = True
        print(all_done)
        vec_env.render()
    env.close()

if __name__ == "__main__":
    main()