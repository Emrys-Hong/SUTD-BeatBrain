"""
A convenient global store for application default values.
This is *not* a config file - more on what to do about that coming here soon.
"""

SAMPLE_RATE = 32768
AUDIO_START = 0
AUDIO_DURATION = None
N_FFT = 1024
HOP_LENGTH = N_FFT // 4
GRIFFINLIM_ITER = 32
PIXELS_PER_CHUNK = 512
TRUNCATE = False
RESAMPLE_TYPE = 'kaiser_best'
WALK_DEPTH = 3
NORMALIZE_AUDIO = True
AUDIO_FORMAT = 'wav'
IMAGE_DROP_TOP = True
IMAGE_FLIP = True
IMAGE_MODE = 'F'
