import config
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

def make_predictions(model: torch.nn.Module, dataset: Dataset, device=None, batch_size=config.BATCH_SIZE, num_workers=0):
    """Make predictions using a model on the test set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    softmax = torch.nn.Softmax(dim=1)
    results = {}

    with torch.inference_mode():
        for batch in loader: # (B, 1, n_mels, time)
            mel = batch["mel"].to(device, non_blocking=True)
            logits = model(mel) # (B, n_classes)
            probs = softmax(logits).cpu().numpy() # (B, n_classes)
            ids = batch["id"] # List of file basenames
            for fid, p in zip(ids, probs):
                results[fid] = p.copy()

    return results

def ensemble_predictions(models, dataset: Dataset, device=None, batch_size=config.BATCH_SIZE, num_workers=0, weights=None):
    """Make predictions by ensembling model predictions on the test set."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    models = list(models)
    n_models = len(models)

    # Model ensemble weights
    if weights is None:
        weights = np.ones(n_models, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() + config.EPS)

    for m in models:
        m.to(device)
        m.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    results = {}

    with torch.inference_mode():
        for batch in loader:
            ids = batch["id"]
            x = batch["mel"].to(device, non_blocking=True)  # (B, 1, C, T)

            agg = None
            for w, m in zip(weights, models):
                logits = m(x) # (B, n_classes)
                logits_cpu = logits.detach().cpu().numpy()
                if agg is None:
                    agg = w * logits_cpu
                else:
                    agg += w * logits_cpu
            exp = np.exp(agg - np.max(agg, axis=1, keepdims=True))
            probs_batch = exp / (exp.sum(axis=1, keepdims=True) + config.EPS)

            for fid, p in zip(ids, probs_batch):
                results[fid] = p.copy()

    return results
