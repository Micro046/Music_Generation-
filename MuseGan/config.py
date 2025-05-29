"""
Configuration file for MuseGAN PyTorch implementation
"""

import torch

# Data parameters
BATCH_SIZE = 64
N_BARS = 2
N_STEPS_PER_BAR = 16
MAX_PITCH = 83
N_PITCHES = MAX_PITCH + 1
N_TRACKS = 4  # Number of tracks in Bach chorales
Z_DIM = 32

# Training parameters
CRITIC_STEPS = 5
GP_WEIGHT = 10
CRITIC_LEARNING_RATE = 0.001
GENERATOR_LEARNING_RATE = 0.001
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
EPOCHS = 6000

# Model loading
LOAD_MODEL = False
MODEL_PATH = "./checkpoint/checkpoint.pth"

# Data path
DATA_PATH = "/content/Jsb16thSeparated.npz"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directories
CHECKPOINT_DIR = "./checkpoint"
OUTPUT_DIR = "./output"
LOGS_DIR = "./logs"