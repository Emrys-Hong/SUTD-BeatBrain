import io
import pathlib
import librosa
from natsort import natsorted
import numpy as np
import tensorflow as tf
from PIL import Image


# region Pre-processing Functions
def load_images_as_tensor(files):
    files = [file.decode('utf8') for file in files.numpy()]
    images = np.asarray([np.asarray(Image.open(file)) for file in files])
    images_tensor = tf.convert_to_tensor(images)
    return images_tensor


def audio_bytes_as_tensor(audio_bytes):
    if isinstance(audio_bytes, tf.Tensor):
        audio_bytes = audio_bytes.numpy()
    file = io.BytesIO(audio_bytes)
    audio, sr = librosa.load(file, sr=32768, res_type='kaiser_fast')
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

    # Reconstruction:
    # spec = librosa.db_to_power(80 * (spec - 1), ref=np.max)
    # audio = librosa.feature.inverse.mel_to_audio(spec, sr=sr, n_fft=n_fft, hop_length=hop_length)
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


# endregion


def load_dataset(data_root, n_fft=4096, hop_length=256, n_mels=512,
                 chunk_size=512, window_size=1, batch_size=1,
                 shuffle=True, prefetch=16, cache=True, parallel=True):
    """
    Given a directory containing audio files, return a `tf.data.Dataset` instance that generates
    spectrogram chunk images.

    Args:
        data_root (str): Root directory containing audio files (and ideally nothing else?)
        n_fft:
        hop_length:
        n_mels:
        chunk_size: Number
        window_size: Number of neighboring spectrogram frames to include in each sample - provides local context
        batch_size: Number of samples per batch
        shuffle (bool): Whether to shuffle the dataset
        prefetch: Number of batches to prefetch in the pipeline. 0 to disable
        cache (bool): Whether to cache the pre-processed dataset to disk
        parallel (bool): Whether to parallelize pre-processing on CPU

    Returns:
        A `tf.data.Dataset` instance
    """
    num_parallel = tf.data.experimental.AUTOTUNE if parallel else None
    data_root = pathlib.Path(data_root).resolve()
    if data_root.is_dir():
        files = natsorted(map(str, filter(pathlib.Path.is_file, data_root.iterdir())))
    else:
        files = [data_root]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(tf.io.read_file)
    dataset = dataset.map(lambda elem: tf.py_function(
        audio_bytes_as_tensor,
        [elem],
        tf.float32
    ), num_parallel_calls=num_parallel)
    dataset = dataset.map(lambda elem: tf.py_function(
        audio_to_spec,
        [elem, n_fft, hop_length, n_mels],
        tf.float32
    ), num_parallel_calls=num_parallel)
    # if cache:
    #     os.makedirs('./tf_cache', exist_ok=True)
    #     dataset = dataset.cache('./tf_cache/cache')
    dataset = dataset.map(lambda elem: tf.py_function(
        split_spec_to_chunks,
        [elem, chunk_size, window_size],
        tf.float32
    ))
    dataset = dataset.unbatch()
    if shuffle:
        dataset = dataset.shuffle(50000)
    if batch_size:
        dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(prefetch)
    return dataset
