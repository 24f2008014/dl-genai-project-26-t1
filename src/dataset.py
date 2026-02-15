import os
import json
import numpy as np
import torch
from audio import load_audio, pad_or_trim, log_mel
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, idx_file, mean, std, mmap=True): # If mmap, will read np arrays from disk instead of loading all into memory
        with open(idx_file, "r") as f:
            self.paths = [l.strip() for l in f if l.strip()]
        self.mmap = mmap
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1,1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1,1)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.mmap:
            npz = np.load(path, mmap_mode="r")
        else:
            npz = np.load(path)
        
        mel = npz["mel"]
        label = int(npz["label"])
        meta = None
        if "meta" in npz:
            try:
                meta = json.loads(str(npz["meta"].tolist()))
            except Exception:
                meta = npz["meta"].tolist()
        
        mel_t = torch.from_numpy(mel).float().unsqueeze(0)
        mel_t = (mel_t - self.mean) / self.std
        return {
            "mel": mel_t,
            "label": label,
            "meta": meta
        }

# Custom collate function for dataloader
def mashup_collate_fn(batch):
    mels = [item["mel"] for item in batch]
    labels = [item["label"] for item in batch]
    meta = [item["meta"] for item in batch]

    mels_tensor = torch.stack(mels, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return {
        "mel": mels_tensor,
        "label": labels_tensor,
        "meta": meta
    }

class TestDataset(Dataset):
    def __init__(self, file_list, mean, std):
        self.files = [str(p) for p in file_list]
        # Reshape to (1, n_mels, 1) so broadcasting with (B, 1, n_mels, T) works
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1)
        self.std  = torch.tensor(std, dtype=torch.float32).view(1, -1, 1)

    def __len__(self):
        return len(self.files)

    def get_mel(self, idx):
        return log_mel(pad_or_trim(load_audio(self.files[idx]))) # Don't peak-normalize y here

    def __getitem__(self, idx):
        mel = self.get_mel(idx) # (n_mels, time)
        mel_t = torch.from_numpy(mel).unsqueeze(0).float() # (1, n_mels, time)
        mel_t = (mel_t - self.mean) / self.std # Normalize (torch broadcasting: (1, n_mels, time) - (1, n_mels, 1))
        file_id = os.path.basename(self.files[idx])
        return {
            "id": file_id,
            "path": self.files[idx],
            "mel": mel_t
        }