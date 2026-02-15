import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from audio import load_audio, pad_or_trim, log_mel

class PrecomputedDataset(Dataset):
    def __init__(self, idx_file, mean, std, transform=None, mmap=True):
        with open(idx_file, "r") as f:
            self.paths = [l.strip() for l in f if l.strip()]
        self.transform = transform
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
        if self.transform is not None:
            mel_t = self.transform(mel_t)
        out = {
            "mel": mel_t,
            "label": label,
            "meta": meta
        }
        return out

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
        # reshape to (1, n_mels, 1) so broadcasting with (B,1,n_mels,T) works
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1)
        self.std  = torch.tensor(std, dtype=torch.float32).view(1, -1, 1)

    def __len__(self):
        return len(self.files)

    def get_mel(self, idx):
        return log_mel(pad_or_trim(load_audio(self.files[idx])))

    def __getitem__(self, idx):
        mel = self.get_mel(idx)                      # (n_mels, time)
        mel_t = torch.from_numpy(mel).unsqueeze(0).float()  # (1, n_mels, time)
        # normalize (torch broadcasting: (1,n_mels,time) - (1,n_mels,1))
        mel_t = (mel_t - self.mean) / self.std
        file_id = os.path.basename(self.files[idx])
        return {
            "id": file_id,
            "path": self.files[idx],
            "mel": mel_t
        }