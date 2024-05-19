import torch
from PPO import PPO
from NeuralNetwork import Convolutional

print(f"PyTorch Version: {torch.__version__}, CUDA Version: {torch.version.cuda}, CUDA Enabled: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(envs, hyperparameters):
	model = PPO(policy_type=Convolutional, envs=envs, **hyperparameters)
	# model.learn(epochs=1000)

hyperparameters = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "clip_ratio": 0.2,
    "render": True,
    "epochs": 1000,
	"device": device,
}