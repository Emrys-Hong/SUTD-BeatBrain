import os
import random
import pathlib
import librosa
from natsort import natsorted
import numpy as np
import tensorflow as tf
from functools import reduce
from PIL import Image

from .. import settings


# region Pre-processing Functions
def load_images_as_tensor(files):
    files = [file.decode('utf8') for file in files.numpy()]
    images = np.asarray([np.asarray(Image.open(file)) for file in files])
    images_tensor = tf.convert_to_tensor(images)
    return images_tensor


def load_audio(path, sample_rate, resample_type):
    if isinstance(path, tf.Tensor):
        path = path.numpy().decode('utf8')
        sample_rate = sample_rate.numpy()
        resample_type = resample_type.numpy().decode('utf8')
    try:
        audio, sr = librosa.load(path, sr=sample_rate, res_type=resample_type)
    except Exception as e:
        print(f"Error loading file '{path}'")
        raise e
    return audio


def audio_to_spec(audio, n_fft, hop_length, n_mels):
    if isinstance(audio, tf.Tensor):
        audio = audio.numpy()
        n_fft = n_fft.numpy()
        hop_length = hop_length.numpy()
        n_mels = n_mels.numpy()
    audio = (audio - audio.mean()) / np.abs(audio).max()
    spec = librosa.feature.melspectrogram(audio, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = spec - spec.min()
    spec = spec / np.abs(spec).max()
    # ==============================================================================
    # RECONSTRUCTION
    # ------------------------------------------------------------------------------
    # spec = librosa.db_to_power(80 * (spec - 1), ref=np.max)
    # audio = librosa.feature.inverse.mel_to_audio(spec, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # ==============================================================================
    return spec


def split_spec_to_chunks(spec, chunk_size, window_size):
    if spec.shape[1] < chunk_size:  # pad if spec is smaller than desired chunk size
        spec = tf.pad(spec, [[0, 0], [0, chunk_size - spec.shape[1]]])
    remainder = spec.shape[1] % chunk_size
    spec = spec[:, :-remainder]
    if spec.shape[1] >= chunk_size:
        sizes = tf.fill([spec.shape[1] // chunk_size], chunk_size)
        chunks = tf.split(spec, sizes, axis=1)
    else:
        chunks = [spec]

    ds = tf.data.Dataset.from_tensor_slices(chunks)
    ds = ds.window(tf.cast(window_size, tf.int64), 1, drop_remainder=True).flat_map(
        lambda x: x.batch(tf.cast(window_size, tf.int64))
    )
    chunks = tf.stack(list(ds))
    return chunks


def load_numpy(path):
    with np.load(path.numpy(), allow_pickle=True) as npz:
        arrays = list(npz.values())
        return np.array(arrays)


def make_windows(array, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(array)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True).flat_map(
        lambda x: x.batch(window_size))
    return dataset


# endregion


def load_dataset(data_root, sample_rate=settings.SAMPLE_RATE, resample_type=settings.RESAMPLE_TYPE,
                 n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH,
                 n_mels=settings.N_MELS, chunk_size=settings.CHUNK_SIZE, channels_last=settings.CHANNELS_LAST,
                 window_size=settings.WINDOW_SIZE, batch_size=settings.BATCH_SIZE,
                 shuffle_buffer=settings.SHUFFLE_BUFFER, prefetch=settings.PREFETCH_DATA,
                 cache=False, data_parallel=settings.DATA_PARALLEL, limit=None):
    """
    Given a directory containing audio files, return a `tf.data.Dataset` instance that generates
    spectrogram chunk images.

    Args:
        data_parallel:
        limit:
        resample_type:
        sample_rate:
        channels_last:
        data_root (str): Root directory containing audio files (and ideally nothing else?)
        n_fft:
        hop_length:
        n_mels:
        chunk_size: Number
        window_size: Number of neighboring spectrogram frames to include in each sample - provides local context
        batch_size: Number of samples per batch
        shuffle_buffer (bool): Whether to shuffle the dataset
        prefetch: Number of batches to prefetch in the pipeline. 0 to disable
        cache (str): Whether to cache the pre-processed dataset to disk

    Returns:
        A `tf.data.Dataset` instance
    """
    num_parallel = settings.NUM_CPUS if data_parallel else None
    data_root = pathlib.Path(data_root).resolve()
    files = natsorted(map(str, filter(pathlib.Path.is_file, data_root.iterdir())))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda path: tf.py_function(
        load_audio,
        [path, sample_rate, resample_type],
        tf.float32
    ), num_parallel_calls=None)
    if cache:  # Caching causes train set to be reused as test set???
        if not isinstance(cache, str):
            cache = ''
        else:
            os.makedirs(cache, exist_ok=True)
        dataset = dataset.cache(filename=cache)
    dataset = dataset.map(lambda audio: tf.py_function(
        audio_to_spec,
        [audio, n_fft, hop_length, n_mels],
        tf.float32
    ), num_parallel_calls=None)
    dataset = dataset.map(lambda spec: tf.py_function(
        split_spec_to_chunks,
        [spec, chunk_size, window_size],
        tf.float32
    ), num_parallel_calls=num_parallel)
    dataset = dataset.unbatch()
    if channels_last:
        dataset = dataset.map(lambda e: tf.transpose(e, perm=[1, 2, 0]), num_parallel_calls=num_parallel)
    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(prefetch)
    if limit:
        dataset = dataset.take(limit)
    return dataset


def load_numpy_dataset(data_root, channels_last=settings.CHANNELS_LAST,
                       window_size=settings.WINDOW_SIZE, batch_size=settings.BATCH_SIZE,
                       shuffle_buffer=settings.SHUFFLE_BUFFER, prefetch=settings.PREFETCH_DATA,
                       data_parallel=settings.DATA_PARALLEL, test_fraction=settings.TEST_FRACTION):
    """
    Given a directory containing audio files, return a `tf.data.Dataset` instance that generates
    spectrogram chunk images.

    Args:
        test_fraction:
        data_parallel:
        channels_last:
        data_root (str): Root directory containing audio files (and ideally nothing else?)
        window_size: Number of neighboring spectrogram frames to include in each sample - provides local context
        batch_size: Number of samples per batch
        shuffle_buffer (bool): Whether to shuffle the dataset
        prefetch: Number of batches to prefetch in the pipeline. 0 to disable

    Returns:
        A `tf.data.Dataset` instance
    """
    num_parallel = tf.data.experimental.AUTOTUNE if data_parallel else None
    data_root = pathlib.Path(data_root).resolve()
    files = list(map(str, filter(pathlib.Path.is_file, data_root.rglob('*.np*'))))
    if shuffle_buffer > 1:
        random.shuffle(files)

    num_test = int(round(test_fraction * len(files)))
    train_test_datasets = [None, None]
    for i in range(2):
        set_files = files[-num_test if i else None:None if i else -num_test]
        dataset = tf.data.Dataset.from_tensor_slices(set_files)
        if shuffle_buffer > 1:
            dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda path: tf.py_function(
            load_numpy,
            [path],
            tf.float32
        ), num_parallel_calls=None)
        dataset = dataset.interleave(lambda chunks: make_windows(chunks, tf.cast(window_size, tf.int64)),
                                     num_parallel_calls=None)
        if channels_last:
            dataset = dataset.map(lambda e: tf.transpose(e, perm=[1, 2, 0]), num_parallel_calls=num_parallel)
        dataset = dataset.batch(batch_size)
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        train_test_datasets[i] = dataset
    train_dataset, test_dataset = train_test_datasets
    return train_dataset, test_dataset
