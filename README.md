# dl-genai-project-26-t1

## Personal Details
- **Student Name**: Manoranjan Kumar
- **Roll Number**: 24f2008014

## Overview
This repository contains all my code for the DL GenAI project for T1 2026. The task for this term was to predict the genre of noisy music mashups, made more challenging by the fact that we had to a) construct our own training dataset using GTZAN song stems and ESC-50 noise samples, and b) ensure our synthesized samples closely resembled the test samples.

The raw datasets (ESC-50, stems, test set) are not included as part of this repository in order to reduce the file size. Please see `README.md` in the `/data` directory for instructions on how to import the data if you have the datasets saved locally.

Please note that this repository uses `uv` for package management, not `pip` or `conda`. `pyproject.toml` and `uv.lock` are present in the root directory, and can be used to install all dependencies automatically.

## Installation
1. Make sure `uv` is installed on your system.
2. Download this repository and `cd` into the root directory.
3. Run `uv sync` to create a virtual environment and automatically install all dependencies.
4. Done. You can now run any script like this, for example: `uv run python -m scripts.synth_full`. Please include the `python -m` bit to install the script as a module before running it, as relative imports won't work properly otherwise :)

## Contents
- `/checkpoints` - Contains the best model checkpoints.
- `/data`
    - `README.md` - Instructions on how to include the raw data.
- `/notebooks`
    - `check_synth_distribution` - 
    - `eda` - Basic exploratory data analysis of the datasets.
    - `kaggle_infer` - The notebook I used to make Kaggle submissions.
    - `milestones/milestone{1..5}.ipynb` - Self-contained notebooks for each of the 5 milestone assignments.
- `/reports`
    - `project_report.pdf` - The full project report.
- `/scripts`
    - `make_predictions.py` - Make predictions on the test set using the best model checkpoints.
    - `run_sweep.py` - Run a hyperparameter sweep using Wandb for logging.
    - `synthesize_mashups.py` - Synthesize training and validation datasets, and compute and save training set statistics.
    - `train_model.py` - Train models with different configurations and log runs using Wandb.
- `/src`
    - `audio.py` - Various audio-related helper functions.
    - `config.py` - Configuration settings for the whole project.
    - `dataset.py` - PyTorch `Dataset` class definitions for train and test, collate functions, and a function to get model embeddings.
    - `deploy.py` - The `app.py` that I used to deploy the project on HuggingFace using Gradio.
    - `infer.py`- Functions for making predictions on the test set.
    - `metrics.py` - Functions for calculating training set statistics and computing FID score.
    - `models.py` - PyTorch model definitions (ResNet, EffNet, ScratchCNN).
    - `synthesis.py` - Functions for generating new synthetic training data from the stems and noise clips.
    - `train.py` - Functions for training/validating one epoch, training a model for multiple epochs, logging runs/sweeps using Wandb, and implementing SpecAugment and mixup for (training) data augmentation.
    - `utils.py` - Utility functions for seeding PyTorch worker RNG and loading the precomputed training set statistics.
- `pyproject.toml` - Project metadata and dependencies (`uv`).
- `uv.lock` - Exact versions of all dependencies (`uv`).