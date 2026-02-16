# This file is `app.py` when deploying on a HF space.
# Also need to include `src/models.py` (same as the file in this repo) in the space repo.
# Also need the 3 model checkpoints and the training stats file in the space repo.

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import librosa

EFFNET_CHECKPOINT = "effnetb0.ckpt"
RESNET_CHECKPOINT = "resnet.ckpt"
SCRATCH_CHECKPOINT = "scratchcnn.ckpt"
TRAIN_STATS = "train_stats.npz"
DEVICE = torch.device("cpu")

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
SR = 22050
TARGET_DURATION = 30.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
EPS = 1e-8

stats = np.load(TRAIN_STATS)
mean = stats["mean"].astype(np.float32)
std = stats["std"].astype(np.float32)
MEAN_T = torch.from_numpy(mean).view(1, -1, 1)
STD_T = torch.from_numpy(std).view(1, -1, 1)

from src.models import EffNetB0, ResNet, ScratchCNN

def setup_models():
    effnet = EffNetB0(num_classes=10, pretrained=False)
    ckpt = torch.load(EFFNET_CHECKPOINT, map_location="cpu", weights_only=False)
    effnet.load_state_dict(ckpt["model_state"])

    resnet = ResNet(num_classes=10, pretrained=False)
    ckpt = torch.load(RESNET_CHECKPOINT, map_location="cpu", weights_only=False)
    resnet.load_state_dict(ckpt["model_state"])

    scratch = ScratchCNN(num_classes=10, in_ch=1, base_ch=32)
    ckpt = torch.load(SCRATCH_CHECKPOINT, map_location="cpu", weights_only=False)
    scratch.load_state_dict(ckpt["model_state"])

    effnet.to(DEVICE).eval()
    resnet.to(DEVICE).eval()
    scratch.to(DEVICE).eval()

    return [effnet, resnet, scratch]

MODELS = setup_models()

def load_audio(path: str, sr=SR):
    """Load an audio file at the specified sampling rate and return the waveform."""
    y, s = librosa.load(path, sr=sr) # y = 1d numpy floating point array, s = sampling rate
    return y.astype(np.float32)

def pad_or_trim(y: np.ndarray, sr=SR, target_duration=TARGET_DURATION):
    """Return a fixed-duration waveform.
    
    If longer than target duration, trim start/end and return the middle.
    
    If shorter, right-pad with zeros.
    """
    target_length = int(sr * target_duration) # Hz * s
    if len(y) >= target_length:
        start = (len(y) - target_length) // 2
        return y[start: start+target_length] # Take the middle bit
    else:
        return np.pad(y, (0, target_length - len(y)), mode="constant", constant_values=0) # Right-pad with 0s

def log_mel(y: np.ndarray, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    """
    Return a log-mel spectrogram for a waveform.

    The amplitude is scaled relative to `ref=1.0`.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # Power-scale mel spectrogram
    # Convert to dB for a log-mel spectrogram
    return librosa.power_to_db(S=S, ref=1.0, amin=1e-10) # ref=1, *not* np.max

def ensemble_predictions(file, weights=None):
    y = pad_or_trim(load_audio(file, sr=SR))
    mel = log_mel(y)
    mel_t = torch.from_numpy(mel).unsqueeze(0).float()
    mel_t = mel_t.unsqueeze(0)
    mel_t = (mel_t - MEAN_T) / STD_T # (B, 1, N_MELS, T)
    mel_t = mel_t.to(DEVICE)

    n_models = len(MODELS)
    if weights is None:
        weights = np.ones(n_models, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    with torch.inference_mode():
        agg = None
        for w, m in zip(weights, MODELS):
            logits = m(mel_t) # (1, C)
            logits_cpu = logits.detach().cpu().numpy()
            if agg is None:
                agg = w * logits_cpu
            else:
                agg += w * logits_cpu
    
    exp = np.exp(agg - np.max(agg, axis=1, keepdims=True))
    probs = exp / (exp.sum(axis=1, keepdims=True) + EPS)
    probs = probs[0]

    top_idx = probs.argsort()[::-1][:3]
    return [(GENRES[i], float(probs[i])) for i in top_idx]

def infer(wav):
    try:
        preds = ensemble_predictions(wav)
        out = [f"{i+1}. {label} (probability {p:.3f})" for i, (label, p) in enumerate(preds)]
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"

title = "DL GenAI Project Jan 26 - Music Genre Classifier"
description = "Upload a .wav file (preferably mono, 22050 Hz) and the classifier will return the top 3 genre predictions."

interface = gr.Interface(
    fn=infer,
    inputs=gr.Audio(sources="upload", type="filepath", label="Upload a .wav", show_label=True),
    outputs="text",
    title=title,
    description=description,
    # allow_flagging="never",
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)