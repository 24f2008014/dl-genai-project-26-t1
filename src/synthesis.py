import json
import os
import random
from pathlib import Path
from functools import partial
from multiprocessing import Pool
import numpy as np
import src.config as config
from src.audio import load_audio, pad_or_trim, add_noise, normalize, log_mel, scale_to_target_rms, time_stretch

def load_stems(song_dirs: list[str]):
    """Return a stacked numpy array of fixed-duration waveforms of the given stems."""
    order = ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    stems = []
    for dir, f in zip(song_dirs, order):
        y = load_audio(Path(dir) / f)
        y = pad_or_trim(y)
        stems.append(y)
    return np.stack(stems, axis=0)
    
def synthesize_sample(
        idx: int,
        out_dir=config.OUT_DIR,
        songs_by_genre=config.SONGS_BY_GENRE,
        noises_range=(2,5),
        snr_policy=[(0.35,(-8,0)), (0.45,(0,6)), (0.20,(6,14))],
        time_stretch_range=(0.95,1.05),
        gain_db_range=(-6,6),
        dominant_stem=None,
        dominant_db_boost=(3,6),
        dominant_db_reduce=(-1,-4),
        save_mix_wave=False,
        seed_base=42,
    ):
    """Synthesize a single song mashup.

    Args:
        idx: Index number for tweaking the RNG for each sample.
        out_dir: Where to save the mashup.
        songs_by_genre: Dictionary mapping genres to all their song stem directories.
        noises_range: Range of how many noises to add.
        snr_policy: SNR range for noises added. First element in tuple is probability, second is range.
        time_stretch_range: Range for applying time-stretch to song stems.
        gain_db_range: Range for randomly applied stem gain.
        dominant_stem: If True, randomly boosts one stem's gain and lowers the others' gains.
        dominant_db_boost: Range of how much to boost the dominant stem's gain by.
        dominant_db_reduce: Range of how much to lower the non-dominant stems' gains by.
        save_mix_wave: If True, also save the raw waveform to the .npz
        seed_base: Base number for seeding the RNG.

    Returns:
        The path to the output .npz file.
    """
    # Pick random genre, songs, number of noises, and noise SNRs
    rng = random.Random(seed_base + idx)
    genre = rng.choice(config.GENRES)
    songs = rng.sample(songs_by_genre[genre], 4)
    n_noises = rng.randint(*noises_range)
    noises = [rng.choice(config.NOISES) for _ in range(n_noises)]
    snrs = []
    for _ in range(n_noises):
        bounds = rng.choices([i[1] for i in snr_policy], [i[0] for i in snr_policy], k=1)[0]
        snrs.append(rng.uniform(*bounds))
    
    # Load stems and time-stretch and alter gain
    stems = load_stems(songs)
    if dominant_stem is not None and rng.uniform(0, 1) < dominant_stem: # Chosen with probability p=dominant_stem
        rnd_idx = rng.randint(0,3) # Random stem to boost gain of
        for i in range(4):
            stems[i] = pad_or_trim(time_stretch(stems[i], rng.uniform(*time_stretch_range))) # First random time-stretch
            if i == rnd_idx:
                gain_db = rng.uniform(*dominant_db_boost) # Boost the gain of the chosen stem
            else:
                gain_db = rng.uniform(*dominant_db_reduce) # Lower the gain of the other stems
            stems[i] = stems[i] * (10 ** (gain_db / 20)) # Apply gain
    else: # Just tweak all stem gains equally randomly
        for i in range(4):
            stems[i] = pad_or_trim(time_stretch(stems[i], rng.uniform(*time_stretch_range)))
            gain_db = rng.uniform(*gain_db_range)
            stems[i] = stems[i] * (10 ** (gain_db / 20))

    # Mix stems together
    mix = np.sum(stems, axis=0)

    # Then add noise(s)
    for noise, snr in zip(noises, snrs):
        mix = add_noise(mix, normalize(load_audio(noise)), snr)

    # Scale the final mix to a target RMS sampled from a normal distribution matching the test set's RMS
    mix = scale_to_target_rms(mix, np.random.normal(config.TEST_RMS_MEAN, config.TEST_RMS_STDEV))

    # Clip to prevent extreme distortion
    mix = np.clip(mix, -1, 1)

    # Generate a log-mel spectrogram
    mel = log_mel(mix)

    # Metadata
    label = config.genre_to_idx[genre]
    out_path = Path(out_dir) / f"{idx:05d}.npz"
    meta = {
        "genre": genre,
        "songs": [str(s) for s in songs], # Paths need to be strings because json.dumps can't process PosixPaths
        "noises": [str(n) for n in noises],
        "snrs": snrs,
        "seed": seed_base + idx
    }
    if save_mix_wave:
        np.savez(file=str(out_path), mel=mel, mix=mix, label=label, meta=json.dumps(meta))
    else:
        np.savez(file=str(out_path), mel=mel, label=label, meta=json.dumps(meta))
    return str(out_path)

def precompute_set(name: str, count: int, processes=8, out_dir=config.OUT_DIR, seed_base=42, **kwargs):
    """Synthesize multiple samples at once, quickly, by using multiple processes."""
    os.makedirs(out_dir, exist_ok=True) # Ex. precomputed/
    out_subdir = Path(out_dir) / name
    os.makedirs(out_subdir, exist_ok=True) # Ex. precomputed/train_v4/
    indices = list(range(count))

    # Fix OUT_DIR and SEED_BASE for all samples
    fn = partial(synthesize_sample, out_dir=out_subdir, seed_base=seed_base, **kwargs)

    # Generate `count` samples using multiple processes simultaneously
    with Pool(processes) as pool:
        for i, out in enumerate(pool.imap(fn, indices), 1):
            if i % 500 == 0:
                print(f"{i}/{count} done. Last saved: {out}")
    
    # Write out an index file once done
    files = sorted([str(p) for p in Path(out_subdir).glob("*.npz")])
    idx_file = Path(out_dir) / f"{name}_list.txt"
    with open(idx_file, "w") as f:
        for p in files:
            f.write(p + "\n")
    print(f"INDEX: {idx_file} ({len(files)} files)")