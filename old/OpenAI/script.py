import retro
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def main():
    # Initialize game environment
    env = retro.make(game='Airstriker-Genesis')
    env = DummyVecEnv([lambda: env])  # Vectorize environment for performance

    # Initialize model with GPU enhancements where applicable
    # tood: for some reason, training on OpenAI's PPO2 algorithm isn't working (the agent isn't doing very well)
    #       unsure if this is a problem with how long the training is occuring or if it's a bad omen lolololol
    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log="./ppo2_airstriker_tensorboard/")

    # Train the model
    model.learn(total_timesteps=10000)  # Adjust timesteps as needed

    # Use trained model
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()