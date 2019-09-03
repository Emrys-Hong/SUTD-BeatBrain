import os
import re
import glob
import time
import numpy as np
import librosa as rosa
import librosa.display
from PIL import Image
from natsort import natsorted
from joblib import Parallel, delayed

from .defaults import *


def walk_dir(root, depth=WALK_DEPTH):
    all_dirs = [root]
    dirs = [root]
    while len(dirs):
        next_dirs = []
        for parent in dirs:
            for f in os.listdir(parent):
                ff = os.path.join(parent, f)
                if os.path.isdir(ff):
                    next_dirs.append(ff)
                    all_dirs.append(ff)
        dirs = next_dirs
    all_dirs = [d for d in all_dirs if d.count(os.path.sep) <= depth]
    return all_dirs


def list_files(root):
    files = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    return files


def load_audio(path, debug=False, **kwargs):
    start = time.time()
    audio, sr = rosa.load(path, **kwargs)
    if debug:
        print(f"Loaded {audio.size / sr:.3f}s of audio at sr={sr} in {time.time() - start:.2f}s")
    return audio, sr


def save_audio(audio, path, sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO, fmt='wav'):
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
        spec_chunkable = spec[:, :last_index]
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


def save_chunk(chunk, path, mode=IMAGE_MODE, remove_top_row=IMAGE_DROP_TOP, flip_vertical=IMAGE_FLIP):
    if remove_top_row:
        chunk = chunk[:-1]
    if flip_vertical:
        chunk = chunk[::-1]
    image = Image.fromarray(chunk, mode=mode)
    image.save(path)


def save_chunks(chunks, output_dir, basename=None, debug=False):
    start = time.time()
    output_dir = os.path.abspath(output_dir)
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
                            res_type=RESAMPLE_TYPE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                            pixels_per_chunk=PIXELS_PER_CHUNK, truncate=TRUNCATE, debug=False):
    audio, sr = load_audio(path, sr=sr, offset=start, duration=duration, res_type=res_type, debug=debug)
    spec = stft(audio, n_fft=n_fft, hop_length=hop_length, debug=debug)
    chunks = spec_to_chunks(spec, pixels_per_chunk=pixels_per_chunk, truncate=truncate, debug=debug)
    save_chunks(chunks, os.path.join(output_dir, os.path.splitext(os.path.basename(path))[0]), debug=debug)
    return chunks


def convert_images_to_audio(paths, output, n_iter=GRIFFINLIM_ITER, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO,
                            fmt=AUDIO_FORMAT, debug=False):
    start = time.time()
    paths = natsorted(paths)
    chunks = load_chunks(paths)
    recon = [griffinlim(chunk, n_iter=n_iter, win_length=n_fft,
                        hop_length=hop_length, debug=False) for chunk in chunks]
    recon = np.concatenate(recon)
    save_audio(recon, output, sr=sr, norm=norm, fmt=fmt)
    if debug:
        print(f"Reconstructed {len(paths)} chunks in {time.time() - start:.2f}s")
    return recon
