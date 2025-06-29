"""
Model-related utility functions for the Architectural Style Identifier application.

This module contains functions for loading the trained fastai model.
Isolating this logic makes the main app cleaner and easier to manage.
"""

import streamlit as st
from pathlib import Path
import platform
import pathlib
from fastai.vision.all import load_learner

@st.cache_resource
def load_learner_and_vocab(model_path_str):
    """
    Loads a fastai learner model from a given path and returns the learner
    and its vocabulary. This function is cached to improve performance.
    """
    model_path = Path(model_path_str)
    print(f"Loading model from: {model_path}")
    try:
        # Platform-specific patch for Windows paths
        if platform.system() == "Windows":
            original_posix_path = pathlib.PosixPath
            try:
                pathlib.PosixPath = pathlib.WindowsPath
                learn = load_learner(model_path, cpu=True)
            finally:
                pathlib.PosixPath = original_posix_path
        else:
            learn = load_learner(model_path, cpu=True)

        vocab = []
        if hasattr(learn.dls, "vocab") and learn.dls.vocab:
            vocab = list(learn.dls.vocab)
            print(f"Model vocabulary loaded: {len(vocab)} classes.")
        else:
            print("Warning: Model vocabulary (dls.vocab) could not be accessed.")

        print("Model loaded successfully.")
        return learn, vocab

    except FileNotFoundError:
        st.error(f"ERROR: Model file '{model_path}' not found.")
        return None, []
    except Exception as e:
        st.error(f"ERROR loading model '{model_path}': {e}")
        import traceback
        traceback.print_exc()
        return None, []
