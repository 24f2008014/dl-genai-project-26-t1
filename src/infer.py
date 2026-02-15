import numpy as np
from dataset import TestDataset
import config
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

def make_predictions(model, dataset, device=None, batch_size=config.BATCH_SIZE, num_workers=0):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    softmax = torch.nn.Softmax(dim=1)
    results = {}

    with torch.inference_mode():
        for batch in loader:
            # (B, 1, n_mels, time)
            mel = batch["mel"].to(device, non_blocking=True)
            logits = model(mel)                         # (B, n_classes)
            probs = softmax(logits).cpu().numpy()      # (B, n_classes)
            ids = batch["id"]                          # list of file basenames
            for fid, p in zip(ids, probs):
                results[fid] = p.copy()

    return results

def ensemble_predictions(models, dataset, device=None, batch_size=config.BATCH_SIZE, num_workers=0, weights=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    models = list(models)
    n_models = len(models)
    if weights is None:
        weights = np.ones(n_models, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / (weights.sum() + config.EPS)

    # Move models to device and set eval
    for m in models:
        m.to(device)
        m.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    results = {}
    i = 1

    with torch.inference_mode():
        for batch in loader:
            print(i * batch_size)
            i += 1
            ids = batch["id"]
            x = batch["mel"].to(device, non_blocking=True)  # (B, 1, C, T)

            agg = None
            for w, m in zip(weights, models):
                logits = m(x)                 # (B, n_classes)
                logits_cpu = logits.detach().cpu().numpy()
                if agg is None:
                    agg = w * logits_cpu
                else:
                    agg += w * logits_cpu
            exp = np.exp(agg - np.max(agg, axis=1, keepdims=True))
            probs_batch = exp / (exp.sum(axis=1, keepdims=True) + config.EPS)

            # store per-id
            for fid, p in zip(ids, probs_batch):
                results[fid] = p.copy()

    return results

effnet_finetuned = None

stats = np.load("mel_stats.npz", mmap_mode="r")
mean = stats["mean"]
std = stats["std"]
test_files = sorted([f for f in Path("data/messy_mashup/mashups").iterdir() if f.is_file()])
test_set = TestDataset(test_files[:10], mean=mean, std=std)
preds = make_predictions(effnet_finetuned, test_set)
rows = []
for id, probs in preds.items():
    genre = config.idx_to_genre[np.argmax(probs)]
    rows.append((id, genre))
df = pd.DataFrame(rows, columns=["id", "genre"])
df["id"] = df["id"].str.extract(r"([0-9]+)")
# df.to_csv("submission.csv", index=False)