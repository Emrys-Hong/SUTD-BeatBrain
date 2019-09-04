import os
import re
import glob
import time
import numpy as np
import librosa
import librosa.display
from PIL import Image
from natsort import natsorted

from .defaults import *


def truepath(path):
    return os.path.abspath(os.path.realpath(os.path.expanduser(os.path.expandvars(path))))


def walk_dir(path, depth=WALK_DEPTH):
    path = truepath(path)
    all_dirs = [path]
    dirs = [path]
    while len(dirs):
        next_dirs = []
        for parent in dirs:
            for f in os.listdir(parent):
                ff = os.path.join(parent, f)
                if os.path.isdir(ff):
                    next_dirs.append(ff)
                    all_dirs.append(ff)
        dirs = next_dirs
    all_dirs = [d for d in all_dirs if os.path.relpath(d, start=path).count(os.path.sep) <= depth]
    return all_dirs


def list_files(path):
    path = truepath(path)
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return files


def load_audio(path, debug=False, **kwargs):
    start = time.time()
    path = truepath(path)
    audio, sr = librosa.load(path, **kwargs)
    if debug:
        print(f"Loaded {audio.size / sr:.3f}s of audio at sr={sr} in {time.time() - start:.2f}s")
    return audio, sr


def save_audio(audio, path, sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO, fmt=AUDIO_FORMAT):
    path = truepath(path)
    if fmt != 'wav':
        raise NotImplementedError("Only .wav is currently supported.")
    librosa.output.write_wav(path, audio, sr, norm=norm)


def stft(audio, debug=False, **kwargs):
    start = time.time()
    spec = np.abs(librosa.stft(audio, **kwargs))
    if debug:
        print(f"STFT'd {audio.size} samples of audio in {time.time() - start:.2f}s")
    return spec / spec.max()


def griffinlim(spec, debug=False, **kwargs):
    start = time.time()
    recon = librosa.griffinlim(spec, **kwargs)
    if debug:
        print(f"Reconstructed {recon.size} samples in {time.time() - start:.2f}s")
    return recon


def spec_to_chunks(spec, pixels_per_chunk=PIXELS_PER_CHUNK, truncate=TRUNCATE, debug=False):
    start = time.time()
    remainder = spec.shape[1] % pixels_per_chunk
    if truncate:
        last_index = spec.shape[1] - remainder
        spec_chunkable = spec[:, :last_index]
    else:
        spec_chunkable = np.pad(spec, ((0, 0), (0, pixels_per_chunk - remainder)), mode='constant')
    if debug:
        if truncate:
            print(f"Truncated spectrogram shape: {spec_chunkable.shape}")
        else:
            print(f"Padded spectrogram shape: {spec_chunkable.shape}")
    if spec_chunkable.shape[1] >= pixels_per_chunk:
        chunks = np.split(spec_chunkable, spec_chunkable.shape[1] // pixels_per_chunk, axis=1)
    else:
        chunks = [spec_chunkable]
    if debug:
        print(f"Split ({spec.shape[1]} x {spec.shape[0]}) image "
              f"into {len(chunks)} chunks in {time.time() - start: .2f}s")
    return chunks


def save_chunk(chunk, path, mode=IMAGE_MODE, remove_top_row=IMAGE_DROP_TOP, flip_vertical=IMAGE_FLIP):
    path = truepath(path)
    if remove_top_row:
        chunk = chunk[:-1]
    if flip_vertical:
        chunk = chunk[::-1]
    image = Image.fromarray(chunk, mode=mode)
    image.save(path)


def save_chunks(chunks, output_dir, basename=None, debug=False):
    start = time.time()
    output_dir = truepath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if basename is None:
        basename = os.path.basename(output_dir)
    if debug:
        print(f"Saving {len(chunks)} chunks to '{output_dir}'...")
    for i, chunk in enumerate(chunks):
        path = os.path.join(output_dir, f"{basename}_{i + 1}.tiff")
        save_chunk(chunk, path)
    if debug:
        print(f"Saved {len(chunks)} chunks in {time.time() - start:.2f}s")


def load_chunks(paths, restore_top_row=IMAGE_DROP_TOP, flip_vertical=IMAGE_FLIP, concatenate=IMAGE_CONCATENATE):
    chunks = []
    for path in paths:
        path = truepath(path)
        chunk = np.asarray(Image.open(path))
        if flip_vertical:
            chunk = chunk[::-1]
        if restore_top_row:
            restored = np.zeros((chunk.shape[0] + 1, *chunk.shape[1:]), dtype=chunk.dtype)
            restored[:chunk.shape[0]] = chunk
            chunk = restored
        chunks.append(chunk)
    if len(chunks) == 1:
        return chunks[0]
    if concatenate:
        chunks = np.concatenate(chunks)
    return chunks


def convert_audio_to_images(path, output_dir, sr=SAMPLE_RATE, start=AUDIO_START, duration=AUDIO_DURATION,
                            res_type=RESAMPLE_TYPE, n_fft=N_FFT, pixels_per_chunk=PIXELS_PER_CHUNK,
                            truncate=TRUNCATE, save=True, debug=False):
    path = truepath(path)
    output_dir = truepath(output_dir)
    audio, sr = load_audio(path, sr=sr, offset=start, duration=duration, res_type=res_type, debug=debug)
    spec = stft(audio, n_fft=n_fft, debug=debug)
    chunks = spec_to_chunks(spec, pixels_per_chunk=pixels_per_chunk, truncate=truncate, debug=debug)
    if save:
        save_chunks(chunks, os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0]), debug=debug)
    return chunks


def convert_images_to_audio(paths, output, n_iter=GRIFFINLIM_ITER, n_fft=N_FFT,
                            sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO,
                            fmt=AUDIO_FORMAT, save=True, debug=False):
    paths = [truepath(path) for path in paths]
    output = truepath(output)
    paths = natsorted(paths)
    chunks = load_chunks(paths)
    recon = [griffinlim(chunk, n_iter=n_iter, win_length=n_fft, debug=debug) for chunk in chunks]
    recon = np.concatenate(recon)
    if save:
        save_audio(recon, output, sr=sr, norm=norm, fmt=fmt)
    return recon
