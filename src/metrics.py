import numpy as np
from pathlib import Path
import src.config as config

def compute_mel_mean_std(mel_paths, n_mels=config.N_MELS):
    sum_bins = np.zeros(n_mels, dtype=np.float64)
    sumsq_bins = np.zeros(n_mels, dtype=np.float64)
    total_frames = 0

    for p in mel_paths:
        mel = np.load(p)["mel"]
        mel = mel.astype(np.float64)
        sum_bins += mel.sum(axis=1)
        sumsq_bins += (mel * mel).sum(axis=1)
        total_frames += mel.shape[1]

    mean = sum_bins / total_frames
    var = (sumsq_bins / total_frames) - (mean * mean)
    std = np.sqrt(np.maximum(var, config.EPS))
    return mean.astype(np.float32), std.astype(np.float32)