from pathlib import Path
import random
import numpy as np
import src.config as config
from src.metrics import compute_mel_mean_std
from src.synthesis import precompute_set

# Split songs for synthesizing train/val, 80/20. No overlaps!
train_songs_by_genre, val_songs_by_genre = {}, {}
for genre, songs in config.SONGS_BY_GENRE.items():
    random.shuffle(songs)
    train_songs_by_genre[genre] = songs[:80]
    val_songs_by_genre[genre]   = songs[80:]

# Synthesize training set (10k samples)
precompute_set(
    name="train_v4",
    count=10000,
    processes=12,
    out_dir=config.OUT_DIR,
    seed_base=42,
    songs_by_genre=train_songs_by_genre,
    noises_range=(2,6),
    snr_policy=[
        (0.35, (-8, 0)), # hard
        (0.45, (0, 6)), # medium
        (0.20, (6, 14)) # soft
    ],
    time_stretch_range=(0.95,1.05),
    gain_db_range=(-6,6),
    dominant_stem=0.3,
    dominant_db_boost=(3,6),
    dominant_db_reduce=(-1,-4),
    save_mix_wave=False
)

# Synthesize validation set (2k samples)
precompute_set(
    name="val_v4",
    count=2000,
    processes=12,
    out_dir=config.OUT_DIR,
    seed_base=42,
    songs_by_genre=val_songs_by_genre,
    noises_range=(2,6),
    snr_policy=[
        (0.35, (-8, 0)), # hard
        (0.45, (0, 6)), # medium
        (0.20, (6, 14)), # soft
    ],
    time_stretch_range=(0.95,1.05),
    gain_db_range=(-6,6),
    dominant_stem=0.3,
    dominant_db_boost=(3,6),
    dominant_db_reduce=(-1,-4),
    save_mix_wave=False
)

# Calculate training mean and standard deviation for normalization later
train = [f for f in Path(f"{config.OUT_DIR}/train_v4").iterdir() if f.is_file()]
mean, std = compute_mel_mean_std(train, config.N_MELS)
np.savez(f"{config.OUT_DIR}/train_stats.npz", mean=mean, std=std)