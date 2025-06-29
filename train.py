# -*- coding: utf-8 -*-
import os
from pathlib import Path
import random
from collections import Counter
import numpy as np
import torch
from fastai.vision.all import *
from functools import partial
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from config import *

# To ensure reproducible results
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

from config import (
    DATA_DIR,
    MODELS_DIR,
    DEFAULT_MODEL_NAME,
    IMG_SIZE,
    BATCH_SIZE,
    VALID_PCT,
    BASE_LEARNING_RATE,
    NUM_FINE_TUNE_EPOCHS,
)

# --- Configuration ---
MODEL_EXPORT_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# --- 0. Prepare Directories ---
print(f"DEBUG: SCRIPT_DIR from config: {SCRIPT_DIR}")
print(f"DEBUG: DATA_DIR from config: {DATA_DIR}")
print(f"DEBUG: MODELS_DIR from config: {MODELS_DIR}")
MODELS_DIR.mkdir(exist_ok=True)

if not DATA_DIR.is_dir():
    print(f"ERROR: Data directory '{DATA_DIR}' not found.")
    exit()

# --- 1. Data Loading and Augmentation (Folder-based) ---
item_tfms = RandomResizedCrop(IMG_SIZE, min_scale=0.75)
batch_tfms = [*aug_transforms(mult=1.0,
                                do_flip=True,
                                flip_vert=False,
                                max_rotate=30.0, # Increased rotation
                                min_zoom=1.0,
                                max_zoom=1.5,   # Increased zoom
                                max_lighting=0.5, # Increased lighting changes
                                max_warp=0.2,   # Increased warping
                                p_affine=0.8,
                                p_lighting=0.8,
                                pad_mode='reflection'
                                ),
              Brightness(max_lighting=0.3, p=0.75, draw=None, batch=False), # Added Brightness
              Contrast(max_lighting=0.3, p=0.75, draw=None, batch=False), # Added Contrast
              RandomErasing(p=0.5, sl=0.02, sh=0.4, min_aspect=0.3), # Added RandomErasing
              Normalize.from_stats(*imagenet_stats)
             ]

try:
    dls = ImageDataLoaders.from_folder(
        DATA_DIR,
        valid_pct=VALID_PCT,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=BATCH_SIZE,
        seed=42,
        num_workers=0 if os.name == 'nt' else min(8, os.cpu_count())
    )
except Exception as e:
    print(f"Error creating ImageDataLoaders: {e}")
    exit()

if len(dls.train_ds) == 0 or len(dls.valid_ds) == 0:
    print(f"No training or validation images were found by ImageDataLoaders. Please check:")
    print(f"1. The structure of the DATA_DIR ('{DATA_DIR}'). It should contain subdirectories for each style.")
    print(f"2. That these subdirectories contain valid image files (jpg, png, etc.).")
    exit()


print(f"DataLoaders created successfully.")
print(f"   Number of classes (styles): {dls.c}")
print(f"   Number of images for training: {len(dls.train_ds)}")
print(f"   Number of images for validation: {len(dls.valid_ds)}")

callbacks_list = [
    SaveModelCallback(monitor='accuracy', comp=np.greater, fname=MODEL_EXPORT_PATH.stem + '_best_during_train', reset_on_fit=True),
    EarlyStoppingCallback(monitor='valid_loss', comp=np.less, min_delta=0.001, patience=5, reset_on_fit=True)
]

print(ARCH)

# Calculate class weights for imbalanced datasets
# Get the labels from the training set file paths
image_labels = [p.parent.name for p in dls.train_ds.items]
class_counts = Counter(image_labels)

# Get the counts in the order of the vocabulary
ordered_counts = [class_counts[label] for label in dls.vocab]

total_samples = len(dls.train_ds)
num_classes = len(dls.vocab)

# Calculate inverse frequency weights
weights = torch.tensor([total_samples / (num_classes * count) if count > 0 else 1 for count in ordered_counts], dtype=torch.float32)
class_weights_tensor = weights.cuda() if torch.cuda.is_available() else weights

learn = vision_learner(
    dls,
    ARCH,
    loss_func=CrossEntropyLossFlat(weight=class_weights_tensor),
    metrics=[accuracy, error_rate, partial(top_k_accuracy, k=3)],
    wd=0.1,
    cbs=callbacks_list
)
learn.path = MODELS_DIR # Set the learner's path to the correct models directory

# --- 3. Find Optimal Learning Rate ---
print("Finding optimal learning rate...")
lr_find_res = learn.lr_find()
suggested_lr = lr_find_res.valley if lr_find_res.valley else lr_find_res.steep if lr_find_res.steep else BASE_LEARNING_RATE

print(f"Suggested learning rate from lr_find: {suggested_lr}")
if suggested_lr < 1e-6:
    print(f"Suggested LR {suggested_lr} is very small, using default BASE_LEARNING_RATE {BASE_LEARNING_RATE}")
    final_lr_to_use = BASE_LEARNING_RATE
else:
    final_lr_to_use = suggested_lr

print(f"Using learning rate: {final_lr_to_use}")

# --- 4. Fine-tuning ---
print("Starting model fine-tuning...")
learn.fine_tune(
    NUM_FINE_TUNE_EPOCHS,
    base_lr=final_lr_to_use, # Use the learning rate found by lr_find
    freeze_epochs=2
)

# --- 5. Export Model ---
learn.export(MODEL_EXPORT_PATH)
print(f"Model has been exported to: {MODEL_EXPORT_PATH}")
print("This '.pkl' file contains the model architecture, weights, and data transformation pipeline.")
