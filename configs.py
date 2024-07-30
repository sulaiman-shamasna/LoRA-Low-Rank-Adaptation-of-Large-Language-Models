import torch

# Architecture
num_features=784
num_hidden_1=128
num_hidden_2=256
num_classes=10

# Hyperparameters
learning_rate=0.005
BATCH_SIZE=64
num_epochs=10

# Settings
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
