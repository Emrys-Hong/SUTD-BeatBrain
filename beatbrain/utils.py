import os
import glob
import time
import numpy as np
import librosa as rosa
import librosa.display
from PIL import Image
from natsort import natsorted
import joblib

from .defaults import *


def walk_dir(root):
    dirs = [root]
    # while we has dirs to scan
    while len(dirs):
        next_dirs = []
        for parent in dirs:
            # scan each dir
            for f in os.listdir(parent):
                # if there is a dir, then save for next iter
                # if it  is a file then yield it (we'll return later)
                ff = os.path.join(parent, f)
                if os.path.isdir(ff):
                    next_dirs.append(ff)
                else:
                    yield ff
        # once we've done all the current dirs then we set up the next iter as the child dirs from the current iter.
        dirs = next_dirs


def load_audio(path, debug=False, **kwargs):
    start = time.time()
    audio, sr = rosa.load(path, **kwargs)
    if debug:
        print(f"Loaded {audio.size / sr:.3f}s of audio at sr={sr} in {time.time() - start:.2f}s")
    return audio, sr


def save_audio(audio, path, sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO, fmt='wav', debug=False, **kwargs):
    if fmt != 'wav':
        raise NotImplementedError("Only .wav is currently supported.")
    rosa.output.write_wav(path, audio, sr, norm=norm)


def stft(audio, debug=False, **kwargs):
    start = time.time()
    spec = np.abs(rosa.stft(audio, **kwargs))
    if debug:
        print(f"STFT'd {audio.size} samples of audio in {time.time() - start:.2f}s")
    return spec / spec.max()


def griffinlim(spec, debug=False, **kwargs):
    start = time.time()
    recon = rosa.griffinlim(spec, **kwargs)
    if debug:
        print(f"Reconstructed {recon.size} samples in {time.time() - start:.2f}s")
    return recon


def spec_to_chunks(spec, pixels_per_chunk=128, truncate=True, debug=False):
    start = time.time()
    remainder = spec.shape[1] % pixels_per_chunk
    if truncate:
        last_index = spec.shape[1] - remainder
        spec_chunkable = spec[..., :last_index]
    else:
        spec_chunkable = np.pad(spec, ((0, 0), (0, pixels_per_chunk - remainder)), mode='constant')
    if debug:
        if truncate:
            print(f"Truncated spectrogram shape: {spec_chunkable.shape}")
        else:
            print(f"Padded spectrogram shape: {spec_chunkable.shape}")
    chunks = np.split(spec_chunkable, spec_chunkable.shape[1] // pixels_per_chunk, axis=1)
    if debug:
        print(f"Split ({spec.shape[1]} x {spec.shape[0]}) image "
              f"into {len(chunks)} chunks in {time.time() - start: .2f}s")
    return chunks


def save_chunks(chunks, output_dir, basename=None, debug=False):
    start = time.time()
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if basename is None:
        basename = os.path.basename(output_dir)
    if debug:
        print(f"Saving {len(chunks)} chunks to '{output_dir}'...")
    for i, chunk in enumerate(chunks):
        img = Image.fromarray(chunk[::-1], mode='F')
        path = os.path.join(output_dir, f"{basename}_{i}.tiff")
        img.save(path)
    if debug:
        print(f"Saved {len(chunks)} chunks in {time.time() - start:.2f}s")


def load_images(path, depth=WALK_DEPTH):
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return Image.open(path)
    elif os.path.isdir(path):
        files = walk_dir(path, depth=depth)


def convert_audio_to_images(path, output_dir, sr=SAMPLE_RATE, start=AUDIO_START, duration=AUDIO_DURATION,
                            res_type=RESAMPLE_TYPE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                            pixels_per_chunk=PIXELS_PER_CHUNK, truncate=TRUNCATE, debug=False):
    audio, sr = load_audio(path, sr=sr, offset=start, duration=duration, res_type=res_type, debug=debug)
    spec = stft(audio, n_fft=n_fft, hop_length=hop_length, debug=debug)
    chunks = spec_to_chunks(spec, pixels_per_chunk=pixels_per_chunk, truncate=truncate, debug=debug)
    save_chunks(chunks, os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0]), debug=debug)
    return chunks


def convert_images_to_audio(path, name_format=IMAGE_NAME_FORMAT,
                            concatenate=CONCATENATE_RECON, n_iter=GRIFFINLIM_ITER, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, debug=False):
    start = time.time()
    files = natsorted(glob.glob(os.path.join(path, name_format)))
    chunks = [np.asarray(Image.open(file))[::-1] for file in files]
    recon = [griffinlim(chunk, n_iter=n_iter, win_length=n_fft,
                        hop_length=hop_length, debug=False) for chunk in chunks]
    if concatenate:
        recon = np.concatenate(recon)
    if debug:
        print(f"Reconstructed {len(files)} chunks in {time.time() - start:.2f}s")
    return recon
