import os
from datetime import datetime
import time

# Training hyperparameters
HYPERPARAMETERS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "total_timesteps": 1e7, #1e7 = 10 million
}

# Directory setup
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODELS_DIR = f"models/PPO-{int(time.time())}"
LOGS_DIR = f"logs/PPO-{int(time.time())}"

# Create directories
for directory in [MODELS_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)