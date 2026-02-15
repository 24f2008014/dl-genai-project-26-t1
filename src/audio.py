import random
import numpy as np
import librosa
import src.config as config
import matplotlib.pyplot as plt

def load_audio(path: str, sr=config.SR):
    """Load an audio file at the specified sampling rate and return the waveform."""
    y, s = librosa.load(path, sr=sr) # y = 1d numpy floating point array, s = sampling rate
    return y.astype(np.float32)

def normalize(y: np.ndarray):
    """Peak-normalize a waveform."""
    return librosa.util.normalize(y).astype(np.float32)

def pad_or_trim(y: np.ndarray, sr=config.SR, target_duration=config.TARGET_DURATION):
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

def rms(y: np.ndarray):
    """Return the RMS value for a waveform."""
    return np.sqrt(np.mean(y**2)+config.EPS)

def rms_db(y: np.ndarray):
    """Return the RMS (dB) value for a waveform."""
    return 20 * np.log10(rms(y))

def scale_to_target_rms(y: np.ndarray, target_db: int):
    """Scale a waveform so its RMS (dB) equals `target_db`."""
    current_db = rms_db(y)
    gain_db = target_db - current_db
    gain_factor = 10.0 ** (gain_db / 20.0)
    return y * gain_factor

def time_stretch(y, rate):
    """Attempt to apply `librosa.effects.time_stretch` to a waveform."""
    if np.abs(rate - 1) < config.EPS: # If rate ~= 1, then don't do anything
        return y
    return librosa.effects.time_stretch(y, rate=rate)

def log_mel(y: np.ndarray, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=config.N_MELS):
    """
    Return a log-mel spectrogram for a waveform.

    The amplitude is scaled relative to `ref=1.0`.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels) # Power-scale mel spectrogram
    # Convert to dB for a log-mel spectrogram
    return librosa.power_to_db(S=S, ref=1.0, amin=1e-10) # ref=1, *not* np.max

def add_noise(mix: np.ndarray, noise: np.ndarray, snr_db):
    """Add a noise to a waveform at a specified SNR, at a random position.
    
    SNR = ratio of desired signal power to noise power. Higher = clearer audio. Lower = more noise.
    """
    if len(noise) >= len(mix):
        start = random.randint(0, len(noise) - len(mix))
        noise_segment = noise[start: start+len(mix)]
        insert_pos = 0
    else:
        insert_pos = random.randint(0, len(mix) - len(noise))
        noise_segment = noise
    
    mix_segment = mix[insert_pos: insert_pos+len(noise_segment)]
    safe_rms = max(rms(mix_segment), rms(mix) * 0.1) # If mix segment is too quiet, fallback to total mix RMS
    desired_noise_rms = safe_rms / (10 ** (snr_db / 20))
    scale = desired_noise_rms / (rms(noise_segment) + config.EPS)
    noise_scaled = noise_segment * scale

    out = mix.copy()
    out[insert_pos: insert_pos+len(noise_scaled)] += noise_scaled
    return out

def display_waveform(y: np.ndarray, sr=config.SR):
    """Plot a waveform using `librosa.display.waveshow`."""
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(y, sr=sr)
    return