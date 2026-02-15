from pathlib import Path

SR = 22050 # Sampling rate
TARGET_DURATION = 30.0 # Output duration, in seconds
N_MELS = 128 # Number of mel bands to generate
N_FFT = 2048 # FFT length
HOP_LENGTH = 512 # Distance between consecutive FFTs
EPS = 1e-8 # Epsilon value to avoid divide-by-zero errors

STEMS_DIR = "data/messy_mashup/genres_stems"
NOISE_DIR = "data/messy_mashup/ESC-50-master/audio"
NOISES = [p for p in Path(NOISE_DIR).iterdir() if p.is_file]
OUT_DIR = "precomputed"
TEST_DIR = "data/messy_mashup/mashups"
CHECKPOINTS_DIR = "checkpoints"
TRAIN_STATS_PATH = "precomputed/train_stats.npz"

WANDB_PROJECT_NAME = "24f2008014-t12026"

GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
SONGS_BY_GENRE = dict()
for g in GENRES:
    SONGS_BY_GENRE[g] = sorted([str(d) for d in (Path(STEMS_DIR)/g).iterdir()])
genre_to_idx = {g: i for i, g in enumerate(GENRES)}
idx_to_genre = {i: g for i, g in enumerate(GENRES)}

# Test set log-mel statistics
TEST_RMS_MEAN = -19.238
TEST_RMS_STDEV = 4.562

# Training params
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4