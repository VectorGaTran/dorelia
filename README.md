# Project Architecture: Architectural Style Identifier

This document outlines the architecture of the Architectural Style Identifier project.

## 1. Project Overview

The project is a web application that allows users to upload an image of a building and have the application identify its architectural style. The application also provides an explanation of which parts of the image were most influential in the model's decision, using Explainable AI (XAI) techniques.

The project is composed of the following main components:

*   **Data Storage**: An `images` directory where training data is stored, organized into subdirectories named by architectural style.
*   **Configuration**: A `config.py` file for centralized project settings.
*   **Model Training**: A primary training script (`train.py`).
*   **Model Utilities**: A `model_utils.py` module for handling model loading.
*   **Prediction Logic**: A `prediction.py` module for encapsulating prediction and XAI generation.
*   **Web Application**: A Streamlit application (`app.py`) that provides the user interface.
*   **Legacy Code**: A `legacy` directory containing older, superseded scripts for historical reference.

## 2. Core Components

### 2.1. `train.py`

*   **Purpose**: This is the primary script for training the image classification model.
*   **Functionality**: It uses the `fastai` library to load image data from the `images/` directory, inferring labels from the subdirectory names. It trains a `resnet50` model using transfer learning, automatically finds an optimal learning rate, and saves the trained model to the `models/` directory. It imports configuration from `config.py`.

### 2.2. `config.py`

*   **Purpose**: Centralized storage for all project-wide constants and configuration settings.
*   **Functionality**: Contains paths to data and models, default model names, and hyperparameters for training. This promotes consistency and ease of modification across the application.

### 2.3. `model_utils.py`

*   **Purpose**: Provides utility functions specifically for loading and managing the machine learning model.
*   **Functionality**: Contains the cached function `load_learner_and_vocab` which loads the `fastai` model and its associated vocabulary (class labels), ensuring efficient model loading within the Streamlit application.

### 2.4. `prediction.py`

*   **Purpose**: Encapsulates the core prediction logic and the generation of Explainable AI (XAI) visualizations.
*   **Functionality**: Contains functions like `get_prediction` to obtain model predictions and `get_xai_visualization` to generate various Grad-CAM and Guided Backpropagation images. It imports low-level XAI utilities from `xai_utils.py`.

### 2.5. `app.py`

*   **Purpose**: This is the main entry point of the Streamlit web application.
*   **Functionality**: It acts as the primary UI controller, setting up the Streamlit page, handling user authentication, managing model selection, and orchestrating calls to `model_utils.py`, `prediction.py`, and `ui_pages.py` to handle specific tasks.

### 2.6. `ui_pages.py`

*   **Purpose**: This module contains the functions that define the content and layout of the different pages within the Streamlit application.
*   **Functionality**: It focuses on rendering the user interface elements for each page (e.g., main analysis page, gallery, map). It now calls functions from `prediction.py` to get model predictions and XAI visualizations, keeping its own logic clean and focused on UI.

### 2.7. `xai_utils.py`

*   **Purpose**: Provides low-level utility functions for generating various types of Explainable AI (XAI) visualizations.
*   **Functionality**: Contains the core implementations for Grad-CAM, Guided Backpropagation, and their combinations. These functions are now primarily called by `prediction.py`.

### 2.8. `legacy/` Directory

*   **Purpose**: Contains older scripts (`train_new.py`, `train_v2.py`, etc.) and database-related utilities (`db_utils.py`, `populate_db.py`) that are no longer in use but are kept for archival purposes.

## 3. Data Flow

1.  **Data Collection**: Training images are organized into subdirectories within the `images/` folder. Each subdirectory's name corresponds to an architectural style.
2.  **Model Training**: The `train.py` script is run. It reads the images and labels from the `images/` directory, trains the classification model, and saves the final `trained_model.pkl` to the `models/` directory.
3.  **Prediction**:
    *   A user accesses the web application (`app.py`).
    *   `app.py` uses `model_utils.py` to load the `trained_model.pkl`.
    *   The user uploads an image through the UI (managed by `ui_pages.py`).
    *   `ui_pages.py` calls `prediction.py` to get the model's prediction and generate XAI explanations.
    *   `prediction.py` utilizes `xai_utils.py` for the underlying XAI computations.
    *   The prediction and the XAI explanation are displayed to the user in the UI.

## 4. Architecture Diagram

```
+------------------------+      +--------------------+
|   images/              |      |      train.py      |
| (organized by style)   |----->|  (Model Training)  |
+------------------------+      +--------------------+
                                         |
                                         |
                                         v
+-----------------+      +------------------+      +-----------------+
|    app.py       |<-----|  model_utils.py  |<-----|   trained_model |
| (Streamlit UI)  |      | (Model Loading)  |      |      (.pkl)     |
|                 |      +------------------+      +-----------------+
|                 |                                         ^
|                 |      +-----------------+                |
|                 |----->|  prediction.py  |----------------+
|                 |      | (Prediction & XAI)|                |
|                 |      +-----------------+                |
|                 |               ^                         |
|                 |               |                         |
|                 |      +-----------------+                |
|                 |----->|   ui_pages.py   |                |
|                 |      | (UI Components) |                |
|                 |      +-----------------+                |
|                 |               ^                         |
|                 |               |                         |
|                 |      +-----------------+                |
|                 |----->|   xai_utils.py  |----------------+
|                 |      | (Low-level XAI) |                |
+-----------------+      +-----------------+                |
                                                            |
                                                            |
                                                            +
                                                            config.py
```

## 5. Key Technologies

*   **Python**: The core programming language.
*   **Streamlit**: For building the interactive web application.
*   **fastai**: For building and training the deep learning model.
*   **grad-cam**: For generating XAI visualizations.
*   **PyTorch**: The underlying deep learning framework for fastai.
*   **Geopy**: For geocoding addresses in the map functionality.
*   **SQLite**: For local user data storage.
*   **Pillow**: For image processing.
*   **OpenCV**: For image manipulation in XAI utilities.
