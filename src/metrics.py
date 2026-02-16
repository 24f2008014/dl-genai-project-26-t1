import numpy as np
import src.config as config
from scipy.linalg import sqrtm

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

def compute_fid(z1, z2):
    """Return the Frechet Inception Distance (FID) score for two distributions."""
    mu1, mu2 = z1.mean(0), z2.mean(0)
    cov1, cov2 = np.cov(z1, rowvar=False), np.cov(z2, rowvar=False)
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - (2 * covmean))