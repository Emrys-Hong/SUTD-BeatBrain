import os
import re
import glob
import time
import numpy as np
import librosa
import librosa.display
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

from . import settings


def truepath(path):
    return settings.os.path.abspath(
        settings.os.path.realpath(settings.os.path.expanduser(settings.os.path.expandvars(path))))


def walk_dir(path, depth=settings.WALK_DEPTH):
    path = truepath(path)
    all_dirs = [path]
    dirs = [path]
    while len(dirs):
        next_dirs = []
        for parent in dirs:
            for f in settings.os.listdir(parent):
                ff = settings.os.path.join(parent, f)
                if settings.os.path.isdir(ff):
                    next_dirs.append(ff)
                    all_dirs.append(ff)
        dirs = next_dirs
    all_dirs = [d for d in all_dirs if settings.os.path.relpath(d, start=path).count(settings.os.path.sep) <= depth]
    return all_dirs


def list_files(path):
    path = truepath(path)
    files = [settings.os.path.join(path, f) for f in settings.os.listdir(path) if
             settings.os.path.isfile(settings.os.path.join(path, f))]
    return files


def load_audio(path, debug=False, **kwargs):
    start = time.time()
    path = truepath(path)
    audio, sr = librosa.load(path, **kwargs)
    if debug:
        print(f"Loaded {audio.size / sr:.3f}s of audio at sr={sr} in {time.time() - start:.2f}s")
    return audio, sr


def save_audio(audio, path, sr=settings.SAMPLE_RATE, norm=settings.NORMALIZE_AUDIO, fmt=settings.AUDIO_FORMAT):
    path = truepath(path)
    if fmt != 'wav':
        raise NotImplementedError("Only .wav is currently supported.")
    librosa.output.write_wav(path, audio, sr, norm=norm)


def griffinlim(spec, debug=False, **kwargs):
    start = time.time()
    recon = librosa.griffinlim(spec, **kwargs)
    if debug:
        print(f"Reconstructed {recon.size} samples in {time.time() - start:.2f}s")
    return recon


def spec_to_chunks(spec, chunk_size, truncate):
    remainder = spec.shape[1] % chunk_size
    if truncate:
        spec = spec[:, :-remainder]
    else:
        spec = np.pad(spec, ((0, 0), (0, chunk_size - remainder)), mode='constant')
    if spec.shape[1] >= chunk_size:
        chunks = np.split(spec, spec.shape[1] // chunk_size, axis=1)
    else:
        chunks = [spec]
    return chunks


def save_chunk(chunk, path, mode=settings.IMAGE_MODE, remove_top_row=settings.IMAGE_DROP_TOP,
               flip_vertical=settings.IMAGE_FLIP):
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
    settings.os.makedirs(output_dir, exist_ok=True)
    if basename is None:
        basename = settings.os.path.basename(output_dir)
    if debug:
        print(f"Saving {len(chunks)} chunks to '{output_dir}'...")
    for i, chunk in enumerate(chunks):
        path = settings.os.path.join(output_dir, f"{basename}_{i + 1}.tiff")
        save_chunk(chunk, path)
    if debug:
        print(f"Saved {len(chunks)} chunks in {time.time() - start:.2f}s")


def load_chunks(paths, restore_top_row=settings.IMAGE_DROP_TOP, flip_vertical=settings.IMAGE_FLIP,
                concatenate=settings.IMAGE_CONCATENATE):
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
    if concatenate:
        chunks = np.concatenate(chunks)
    return chunks


def convert_audio_to_arrays(inp, out_dir, sr=settings.SAMPLE_RATE, offset=settings.AUDIO_START,
                            duration=settings.AUDIO_DURATION, n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH,
                            n_mels=settings.N_MELS, chunk_size=settings.CHUNK_SIZE, skip=0):
    inp = Path(inp)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise ValueError(f"Input must be a valid file or directory. Got '{inp}'")
    elif inp.is_dir():
        paths = natsorted(filter(Path.is_file, inp.rglob('*')))
    else:
        paths = [inp]
    print(f"Converting files in '{inp}' to Numpy arrays...")
    print(f"Arrays will be saved in '{out_dir}'.")
    with tqdm(paths) as pbar:
        for i, path in enumerate(pbar):
            if i < skip:
                continue
            path = Path(path)
            output = out_dir.joinpath(path.relative_to(inp))
            output = output.parent.joinpath(output.stem)
            audio, sr = librosa.load(path, sr=sr, offset=offset, duration=duration, res_type=settings.RESAMPLE_TYPE)
            audio -= audio.mean()
            audio /= np.abs(audio).max()
            spec = librosa.feature.melspectrogram(audio, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            spec = librosa.power_to_db(spec, ref=np.max)
            spec = spec - spec.min()
            spec = spec / np.abs(spec).max()
            chunks = spec_to_chunks(spec, chunk_size, settings.TRUNCATE)
            pbar.write(f"Converting '{path}'...")
            output.mkdir(parents=True, exist_ok=True)
            for i, chunk in enumerate(chunks):
                np.save(output.joinpath(str(i)).resolve(), chunk)


def convert_audio_to_images(path, output_dir, sr=settings.SAMPLE_RATE, start=settings.AUDIO_START,
                            duration=settings.AUDIO_DURATION,
                            res_type=settings.RESAMPLE_TYPE, n_fft=settings.N_FFT, chunk_size=settings.CHUNK_SIZE,
                            truncate=settings.TRUNCATE, save=True, debug=False):
    path = truepath(path)
    output_dir = truepath(output_dir)
    audio, sr = load_audio(path, sr=sr, offset=start, duration=duration, res_type=res_type, debug=debug)
    spec = librosa.stft(audio, n_fft=n_fft)
    chunks = spec_to_chunks(spec, chunk_size, truncate)
    if save:
        save_chunks(chunks,
                    settings.os.path.join(output_dir, settings.os.path.splitext(settings.os.path.basename(path))[0]),
                    debug=debug)
    return chunks


def convert_images_to_audio(paths, output, n_iter=settings.GRIFFINLIM_ITER, n_fft=settings.N_FFT,
                            sr=settings.SAMPLE_RATE, norm=settings.NORMALIZE_AUDIO,
                            fmt=settings.AUDIO_FORMAT, save=True, debug=False):
    paths = [truepath(path) for path in paths]
    output = truepath(output)
    paths = natsorted(paths)
    chunks = load_chunks(paths)
    recon = [griffinlim(chunk, n_iter=n_iter, win_length=n_fft, debug=debug) for chunk in chunks]
    recon = np.concatenate(recon)
    if save:
        save_audio(recon, output, sr=sr, norm=norm, fmt=fmt)
    return recon
