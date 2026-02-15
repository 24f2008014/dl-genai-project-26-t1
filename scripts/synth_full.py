import numpy as np
from pathlib import Path
from src.metrics import compute_mel_mean_std
import src.config as config
import random
from src.synthesis import precompute_set

train_songs_by_genre, val_songs_by_genre = {}, {}
for genre, songs in config.SONGS_BY_GENRE.items():
    random.shuffle(songs)
    train_songs_by_genre[genre] = songs[:80]
    val_songs_by_genre[genre]   = songs[80:]

precompute_set(
    "train_v4",
    count=10000,
    processes=12,
    out_dir=config.OUT_DIR,
    seed_base=42,
    songs_by_genre=train_songs_by_genre,
    snr_range=(-3,6),
    noises_range=(2,6),
    time_stretch_range=(0.95,1.05),
    gain_db_range=(-6,6),
    save_mix_wave=False,
    dominant_stem=0.3,
    dominant_db_boost=(3,6),
    dominant_db_reduce=(-1,-4),
    snr_mixture=True,
    snr_policy=[
        (0.35, (-8, 0)),   # hard
        (0.45, (0, 6)),    # medium
        (0.20, (6, 14)),   # soft
    ],
)

precompute_set(
    "val_v4",
    count=2000,
    processes=12,
    out_dir=config.OUT_DIR,
    seed_base=42,
    songs_by_genre=val_songs_by_genre,
    snr_range=(-3,6),
    noises_range=(2,6),
    time_stretch_range=(0.95,1.05),
    gain_db_range=(-6,6),
    save_mix_wave=False,
    dominant_stem=0.3,
    dominant_db_boost=(3,6),
    dominant_db_reduce=(-1,-4),
    snr_mixture=True,
    snr_policy=[
        (0.35, (-8, 0)),   # hard
        (0.45, (0, 6)),    # medium
        (0.20, (6, 14)),   # soft
    ],
)

synth = [f for f in Path("precomputed/train_v4").iterdir() if f.is_file()]
mean, std = compute_mel_mean_std(synth, config.N_MELS)
np.savez("mel_stats_new.npz", mean=mean, std=std)