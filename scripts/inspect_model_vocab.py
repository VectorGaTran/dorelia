# -*- coding: utf-8 -*
import os
from pathlib import Path
from fastai.vision.all import load_learner

# --- Configuration ---
# Get the absolute path of the directory containing the script
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTANT ---
# Set the name of the model file you want to inspect.
# This defaults to the model created by train_v3.py
MODEL_NAME = "style_model_fastai_resnet50_v3_folder_labeled_lrf.pkl"
MODEL_PATH = SCRIPT_DIR / "models" / MODEL_NAME

def inspect_vocab(model_path):
    """
    Loads a fastai learner and prints its vocabulary (the classes it was trained on).
    """
    if not model_path.exists():
        print(f"Error: Model file not found at \"{model_path}\"")
        print("Please make sure you have trained the model and the MODEL_NAME is correct.")
        return

    try:
        print(f"Loading model from: {model_path}")
        # The CPU=True argument is useful if you are running this on a machine
        # without a GPU, but it will work fine even if you have one.
        learn = load_learner(model_path, cpu=True)
    except Exception as e:
        print(f"An error occurred while loading the learner: {e}")
        return

    # Access the vocabulary from the DataLoaders object
    vocab = learn.dls.vocab

    print("\n----------------------------------------")
    print("         Model Vocabulary         ")
    print("----------------------------------------")
    print(f"The model was trained to recognize the following {len(vocab)} classes:")
    
    # Print labels, one per line for clarity
    for i, label in enumerate(vocab):
        print(f"{i+1}. {label}")
        
    print("----------------------------------------\n")

if __name__ == '__main__':
    inspect_vocab(MODEL_PATH)
