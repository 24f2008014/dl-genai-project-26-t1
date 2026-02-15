from pathlib import Path
import os
from functools import partial
from multiprocessing import Pool
import numpy as np
import random
import config
import json
from audio import load_audio, pad_or_trim, add_noise, normalize, log_mel, scale_to_target_rms, time_stretch

def load_stems(song_dirs: list[str]):
    """Return a stacked array of fixed-duration waveforms of the given stems."""
    order = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    stems = []
    for dir, f in zip(song_dirs, order):
        y = load_audio(Path(dir) / f)
        y = pad_or_trim(y)
        stems.append(y)
    return np.stack(stems, axis=0)
    
def synthesize_sample(
        idx,
        out_dir=config.OUT_DIR,
        songs_by_genre=config.SONGS_BY_GENRE,
        snr_range=(-3,6),
        noises_range=(2,5),
        time_stretch_range=(0.95,1.05),
        gain_db_range=(-6,6),
        seed_base=42,
        save_mix_wave=False,
        dominant_stem=None,
        dominant_db_boost=(3,6),
        dominant_db_reduce=(-1,-4),
        snr_mixture=False,
        snr_policy=[
            (0.35, (-8, 0)),   # hard
            (0.45, (0, 6)),    # medium
            (0.20, (6, 14)),   # soft
        ],
    ):
    rng = random.Random(seed_base + idx)
    genre = rng.choice(config.GENRES)
    songs = rng.sample(songs_by_genre[genre], 4)
    n_noises = rng.randint(*noises_range)
    noises = [rng.choice(config.NOISES) for _ in range(n_noises)]
    snrs = []
    if snr_mixture:
        for _ in range(n_noises):
            bounds = rng.choices([i[1] for i in snr_policy], [i[0] for i in snr_policy], k=1)[0]
            snrs.append(rng.uniform(*bounds))
    else:
        snrs = [rng.uniform(*snr_range) for _ in range(n_noises)]
    
    stems = load_stems(songs)
    if dominant_stem is not None and rng.uniform(0, 1) < dominant_stem:
        rnd_idx = rng.randint(0,3)
        for i in range(stems.shape[0]):
            stems[i] = pad_or_trim(time_stretch(stems[i], rng.uniform(*time_stretch_range)))
            if i == rnd_idx:
                gain_db = rng.uniform(*dominant_db_boost)
            else:
                gain_db = rng.uniform(*dominant_db_reduce)
            stems[i] = stems[i] * (10 ** (gain_db / 20))
    else:
        for i in range(stems.shape[0]):
            stems[i] = pad_or_trim(time_stretch(stems[i], rng.uniform(*time_stretch_range)))
            gain_db = rng.uniform(*gain_db_range)
            stems[i] = stems[i] * (10 ** (gain_db / 20))

    mix = np.sum(stems, axis=0)
    for noise, snr in zip(noises, snrs):
        mix = add_noise(mix, normalize(load_audio(noise)), snr)

    mix = scale_to_target_rms(mix, np.random.normal(-19.238, 4.562))
    mix = np.clip(mix, -1, 1)
    mel = log_mel(mix)
    label = config.genre_to_idx[genre]
    out_path = Path(out_dir) / f"{idx:05d}.npz"
    meta = {
        "genre": genre,
        "songs": [str(s) for s in songs],
        "noises": [str(n) for n in noises],
        "snrs": snrs,
        "seed": seed_base + idx
    }

    if save_mix_wave:
        np.savez(
            file=str(out_path),
            mel=mel,
            mix=mix,
            label=label,
            meta=json.dumps(meta)
        )
    else:
        np.savez(
            file=str(out_path),
            mel=mel,
            label=label,
            meta=json.dumps(meta)
        )
        
    return str(out_path)

def precompute_set(name: str, count: int, processes=8, out_dir=config.OUT_DIR, seed_base=42, **kwargs):
    os.makedirs(out_dir, exist_ok=True)
    out_subdir = Path(out_dir) / name
    os.makedirs(out_subdir, exist_ok=True)
    indices = list(range(count))
    fn = partial(synthesize_sample, out_dir=out_subdir, seed_base=seed_base, **kwargs)

    with Pool(processes) as pool:
        for i, out in enumerate(pool.imap(fn, indices), 1):
            if i % 500 == 0:
                print(f"{i}/{count} done. Last saved: {out}")

    files = sorted([str(p) for p in Path(out_subdir).glob("*.npz")])
    idx_file = Path(out_dir) / f"{name}_list.txt"
    with open(idx_file, "w") as f:
        for p in files:
            f.write(p + "\n")
    print(f"INDEX: {idx_file} ({len(files)} files)")