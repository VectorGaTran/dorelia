"""
Centralized configuration file for the Architectural Style Identifier application.

This file contains all the paths, model names, and other constants used across the project.
This makes it easy to update settings without having to search through multiple scripts.
"""

from pathlib import Path
import os

# --- Core Paths ---

# Get the absolute path of the directory containing this config file
# This makes all other paths relative to the project root, ensuring they work correctly.
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Directory where training images are stored, organized by style.
DATA_DIR = SCRIPT_DIR / "images"

# Directory where trained model files (.pkl) are saved.
MODELS_DIR = SCRIPT_DIR / "models"


# --- Model Configuration ---

ARCH="resnet50"

# The default model file to be used by the application.
# This is the model that `scripts/train.py` produces.
DEFAULT_MODEL_NAME = f"style_model_fastai_{ARCH}_v4_folder_labeled_lrf.pkl"

# The full path to the default model file.
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME


# --- Training Script Hyperparameters ---
# These values are used by `scripts/train.py`.

IMG_SIZE = 224
BATCH_SIZE = 32
VALID_PCT = 0.2  # 20% of the data will be used for validation

# A fallback learning rate if the learning rate finder fails.
BASE_LEARNING_RATE = 1e-3

# Number of epochs for the fine-tuning phase.
NUM_FINE_TUNE_EPOCHS = 30
