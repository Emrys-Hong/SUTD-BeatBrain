"""
A convenient global store for application default values.
This is *not* a config file - more on what to do about that coming here soon.
"""
import os
import pathlib
from datetime import datetime

# Data Locations
ROOT_DIR = str(pathlib.Path(__file__).parents[1].absolute())  # Project root
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "data/array/pendulum")  # Directories containing audio files
OUTPUT_DIR = os.path.join(ROOT_DIR, "data/output")  # Generated files (audio, images, etc.) go here
LOG_DIR = os.path.join(ROOT_DIR, "data/logs")  # Python logs go here
TENSORBOARD_DIR = os.path.join(ROOT_DIR, "data/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
MODEL_WEIGHTS = os.path.join(ROOT_DIR, 'models', 'cvae_pendulum.h5')

# ========================================================================
# Don't modify anything below this line unless you know what you're doing.
# ========================================================================
# Program Defaults - Best not to touch these
SAMPLE_RATE = 32768  # Audio files are resampled to this many samples per second
RESAMPLE_TYPE = 'kaiser_fast'  # Resampling algorithm used by Librosa
N_FFT = 4096  # STFT window size (in samples)
HOP_LENGTH = 256  # STFT stride length (in samples)
N_MELS = 512  # Number of frequency bins per frame (timestep)
EXAMPLES_TO_GENERATE = 16
NUM_CPUS = int(round(os.cpu_count() * 0.75))  # Use ~60% of available CPUs

# Default Hyperparameters
EPOCHS = 100
LATENT_DIMS = 32

# Data Options
TEST_FRACTION = 0.2
CHUNK_SIZE = 512  # Number of frames (timesteps) per spectrogram chunk
CHANNELS_LAST = True
WINDOW_SIZE = 1  # Number of spectrogram chunks per window
BATCH_SIZE = 8  # Number of windows per data sample
SHUFFLE_BUFFER = 1024  # Buffer size for shuffling data samples
PREFETCH_DATA = 128  # Data samples to prefetch (resource intensive, but uses GPU more efficiently)
DATA_PARALLEL = True  # Parallelize CPU-intensive data loading (resource intensive)

# CLI Defaults
AUDIO_START = 0
AUDIO_DURATION = None
GRIFFINLIM_ITER = 32
TRUNCATE = False
WALK_DEPTH = 5
NORMALIZE_AUDIO = True
AUDIO_FORMAT = 'wav'
IMAGE_DROP_TOP = True
IMAGE_FLIP = True
IMAGE_MODE = 'F'
IMAGE_CONCATENATE = False
