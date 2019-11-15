import warnings
warnings.simplefilter("ignore", UserWarning)
import enum
import numpy as np
import librosa
import imageio
import soundfile as sf
from pathlib import Path
from natsort import natsorted
from colorama import Fore
from tqdm import tqdm
from audioread.exceptions import DecodeError

from . import settings


# region Data Types
class DataType(enum.Enum):
    AUDIO = 1
    NUMPY = 2
    ARRAY = 2
    IMAGE = 3
    UNKNOWN = 4
    AMBIGUOUS = 5
    ERROR = 6


EXTENSIONS = {
    DataType.AUDIO: ['wav', 'flac', 'mp3', 'ogg'],  # TODO: Remove artificial limit on supported audio formats
    DataType.NUMPY: ['npy', 'npz'],
    DataType.IMAGE: ['exr']
}


def get_data_type(path, raise_exception=False):
    """
    Given a file or directory, return the (homogeneous) data type contained in that path.

    Args:
        path: Path at which to check the data type.
        raise_exception: Whether to raise an exception on unknown or ambiguous data types.

    Returns:
        DataType: The type of data contained at the given path (Audio, Numpy, or Image)

    Raises:
        ValueError: If `raise_exception` is True, the number of matched data types is either 0 or >1.
    """
    print(f"Checking input type(s) in {Fore.YELLOW}'{path}'{Fore.RESET}...")
    found_types = set()
    path = Path(path)
    files = []
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = filter(Path.is_file, path.rglob('*'))
    for f in files:
        for dtype, exts in EXTENSIONS.items():
            suffix = f.suffix[1:]
            if suffix in exts:
                found_types.add(dtype)
    if len(found_types) == 0:
        dtype = DataType.UNKNOWN
        if raise_exception:
            raise ValueError(f"Unknown source data type. No known file types we matched.")
    elif len(found_types) == 1:
        dtype = found_types.pop()
    else:
        dtype = DataType.AMBIGUOUS
        if raise_exception:
            raise ValueError(f"Ambiguous source data type. The following types were matched: {found_types}")
    print(f"Determined input type to be {Fore.CYAN}'{dtype.name}'{Fore.RESET}")
    return dtype


# endregion

# region Helper functions
def get_paths(inp, parents=False, sort=True):
    """
    Recursively get the filenames under a given path

    Args:
        inp (str): The path to search for files under
        parents (bool): If True, return the unique parent directories of the found files
    """
    inp = Path(inp)
    if not inp.exists():
        raise ValueError(f"Input must be a valid file or directory. Got '{inp}'")
    elif inp.is_dir():
        paths = filter(Path.is_file, inp.rglob('*'))
        if parents:
            paths = {p.parent for p in paths}  # Unique parent directories
        paths = natsorted(paths) if sort else list(paths)
    else:
        paths = [inp]
    return paths


def split_spectrogram(spec, chunk_size, truncate=True, axis=1):
    """
    Split a numpy array along the chosen axis into fixed-length chunks

    Args:
        spec (np.ndarray): The array to split along the chosen axis
        chunk_size (int): The number of elements along the chosen axis in each chunk
        truncate (bool): If True, the array is truncated such that the number of elements
                         along the chosen axis is a multiple of `chunk_size`.
                         Otherwise, the array is zero-padded to a multiple of `chunk_size`.

    Returns:
        list: A list of arrays of equal size
    """
    if spec.shape[axis] >= chunk_size:
        remainder = spec.shape[axis] % chunk_size
        if truncate:
            spec = spec[:, :-remainder]
        else:
            spec = np.pad(spec, ((0, 0), (0, chunk_size - remainder)), mode='constant')
        chunks = np.split(spec, spec.shape[axis] // chunk_size, axis=axis)
    else:
        chunks = [spec]
    return chunks


def load_images(path, flip=True):
    """
    Load a sequence of spectrogram images from a directory as arrays

    Args:
        path: The directory to load images from
        flip (bool): Whether to flip the images vertically
    """
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        files = []
        for ext in EXTENSIONS[DataType.IMAGE]:
            files.extend(path.glob(f'*.{ext}'))
        files = natsorted(files)
    chunks = [imageio.imread(file) for file in files]
    if flip:
        chunks = [chunk[::-1] for chunk in chunks]
    return chunks


def load_arrays(path):
    """
    Load a sequence of spectrogram arrays from a npy or npz file

    Args:
        path: The file to load arrays from
    """
    with np.load(path) as npz:
        keys = natsorted(npz.keys())
        chunks = [npz[k] for k in keys]
        return chunks


def audio_to_spectrogram(audio, normalize=False, norm_kwargs={}, **kwargs):
    """
    Convert an array of audio samples to a mel spectrogram

    Args:
        audio (np.ndarray): The array of audio samples to convert
        normalize (bool): Whether to log and normalize the spectrogram to [0, 1] after conversion
        norm_kwargs (dict): Additional keyword arguments to pass to the spectrogram normalization function
    """
    spec = librosa.feature.melspectrogram(audio, **kwargs)
    if normalize:
        spec = normalize_spectrogram(spec, **norm_kwargs)
    return spec


def normalize_spectrogram(spec, top_db=settings.TOP_DB, ref=np.max, **kwargs):
    """
    Log and normalize a mel spectrogram using `librosa.power_to_db()`
    """
    return (librosa.power_to_db(spec, top_db=top_db, ref=ref, **kwargs) / top_db) + 1


def spectrogram_to_audio(spec, denormalize=False, norm_kwargs={}, **kwargs):
    """
    Convert a mel spectrogram to audio

    Args:
        spec (np.ndarray): The mel spectrogram to convert to audio
        denormalize (bool): Whether to exp and denormalize the spectrogram before conversion
        norm_kwargs (dict): Additional keyword arguments to pass to the spectrogram denormalization function
    """
    if denormalize:
        spec = denormalize_spectrogram(spec, **norm_kwargs)
    audio = librosa.feature.inverse.mel_to_audio(spec, **kwargs)
    return audio


def denormalize_spectrogram(spec, top_db=settings.TOP_DB, ref=32768, **kwargs):
    """
    Exp and denormalize a mel spectrogram using `librosa.db_to_power()`
    """
    return librosa.db_to_power((spec - 1) * top_db, ref=ref)


def save_arrays(chunks, output, compress=True):
    """
    Save a sequence of arrays to a npy or npz file.

    Args:
        chunks (list): A sequence of arrays to save
        output (str): The file to save thethe arrays to'
        compress (bool): Whether to use `np.savez` to compress the output file
    """
    save = np.savez_compressed if compress else np.savez
    save(str(output), *chunks)


def save_chunks_image(chunks, output, flip):
    for j, chunk in enumerate(chunks):
        if flip:
            chunk = chunk[::-1]
        imageio.imwrite(output.joinpath(f"{j}.exr"), chunk)


def get_numpy_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def get_image_output_path(path, out_dir, inp):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.stem)
    output.mkdir(parents=True, exist_ok=True)
    return output


def get_audio_output_path(path, out_dir, inp, fmt):
    path = Path(path)
    out_dir = Path(out_dir)
    inp = Path(inp)
    output = out_dir.joinpath(path.relative_to(inp))
    output = output.parent.joinpath(output.name).with_suffix(f'.{fmt}')
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


# endregion

# region Converters
def convert_audio_to_numpy(inp, out_dir, sr=settings.SAMPLE_RATE, offset=settings.AUDIO_OFFSET,
                           duration=settings.AUDIO_DURATION, res_type=settings.RESAMPLE_TYPE,
                           n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH,
                           n_mels=settings.N_MELS, chunk_size=settings.CHUNK_SIZE,
                           truncate=settings.TRUNCATE, skip=0):
    paths = get_paths(inp, parents=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to Numpy arrays...")
    print(f"Arrays will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        try:
            audio, sr = librosa.load(str(path), sr=sr, offset=offset, duration=duration, res_type=res_type)
        except DecodeError as e:
            print(f"Error decoding {path}: {e}")
            continue
        spec = audio_to_spectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, normalize=True)
        chunks = split_spectrogram(spec, chunk_size, truncate=truncate)
        output = get_numpy_output_path(path, out_dir, inp)
        save_arrays(chunks, output)


def convert_image_to_numpy(inp, out_dir, flip=settings.IMAGE_FLIP, skip=0):
    paths = get_paths(inp, parents=True)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to Numpy arrays...")
    print(f"Arrays will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_images(path, flip=flip)
        output = get_numpy_output_path(path, out_dir, inp)
        save_arrays(chunks, output)


def convert_audio_to_image(inp, out_dir, sr=settings.SAMPLE_RATE, offset=settings.AUDIO_OFFSET,
                           duration=settings.AUDIO_DURATION, res_type=settings.RESAMPLE_TYPE,
                           n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH, n_mels=settings.N_MELS,
                           chunk_size=settings.CHUNK_SIZE, truncate=settings.TRUNCATE,
                           flip=settings.IMAGE_FLIP, skip=0):
    paths = get_paths(inp, parents=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to images...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        try:
            audio, sr = librosa.load(str(path), sr=sr, offset=offset, duration=duration, res_type=res_type)
        except DecodeError as e:
            print(f"Error decoding {path}: {e}")
            continue
        spec = audio_to_spectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, normalize=True)
        chunks = split_spectrogram(spec, chunk_size, truncate=truncate)
        output = get_image_output_path(path, out_dir, inp)
        save_chunks_image(chunks, output, flip)


def convert_numpy_to_image(inp, out_dir, flip=settings.IMAGE_FLIP, skip=0):
    paths = get_paths(inp, parents=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to images...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_arrays(path)
        output = get_image_output_path(path, out_dir, inp)
        save_chunks_image(chunks, output, flip)


def convert_numpy_to_audio(inp, out_dir, sr=settings.SAMPLE_RATE, n_fft=settings.N_FFT,
                           hop_length=settings.HOP_LENGTH, fmt=settings.AUDIO_FORMAT,
                           offset=settings.AUDIO_OFFSET, duration=settings.AUDIO_DURATION, skip=0):
    paths = get_paths(inp, parents=False)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to audio...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_arrays(path)
        audio = spectrogram_to_audio(np.concatenate(chunks, axis=-1), sr=sr, n_fft=n_fft, hop_length=hop_length, denormalize=True)
        output = get_audio_output_path(path, out_dir, inp, fmt)
        sf.write(output, audio, sr)


def convert_image_to_audio(inp, out_dir, sr=settings.SAMPLE_RATE, n_fft=settings.N_FFT,
                           hop_length=settings.HOP_LENGTH, fmt=settings.AUDIO_FORMAT,
                           offset=settings.AUDIO_OFFSET, duration=settings.AUDIO_DURATION,
                           flip=settings.IMAGE_FLIP, skip=0):
    paths = get_paths(inp, parents=True)
    print(f"Converting files in {Fore.YELLOW}'{inp}'{Fore.RESET} to audio...")
    print(f"Images will be saved in {Fore.YELLOW}'{out_dir}'{Fore.RESET}\n")
    for i, path in enumerate(tqdm(paths, desc="Converting")):
        if i < skip:
            continue
        tqdm.write(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET}...")
        chunks = load_images(path, flip=flip)
        audio = spectrogram_to_audio(np.concatenate(chunks, axis=-1), sr=sr, n_fft=n_fft, hop_length=hop_length, denormalize=True)
        output = get_audio_output_path(path, out_dir, inp, fmt)
        sf.write(output, audio, sr)


# endregion

# region Functions used by the `click` CLI
def convert_to_numpy(inp, out_dir, sr=settings.SAMPLE_RATE, offset=settings.AUDIO_OFFSET,
                     duration=settings.AUDIO_DURATION, res_type=settings.RESAMPLE_TYPE,
                     n_fft=settings.N_FFT, hop_length=settings.HOP_LENGTH,
                     n_mels=settings.N_MELS, chunk_size=settings.CHUNK_SIZE, truncate=settings.TRUNCATE,
                     flip=settings.IMAGE_FLIP, skip=0):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_numpy(inp, out_dir, sr=sr, offset=offset, duration=duration, res_type=res_type,
                                      n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                      chunk_size=chunk_size, truncate=truncate, skip=skip)
    elif dtype == DataType.IMAGE:
        return convert_image_to_numpy(inp, out_dir, flip=flip, skip=skip)


def convert_to_image(inp, out_dir, sr=settings.SAMPLE_RATE, offset=settings.AUDIO_OFFSET,
                     duration=settings.AUDIO_DURATION, res_type=settings.RESAMPLE_TYPE, n_fft=settings.N_FFT,
                     hop_length=settings.HOP_LENGTH, chunk_size=settings.CHUNK_SIZE,
                     truncate=settings.TRUNCATE, flip=settings.IMAGE_FLIP, skip=0):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.AUDIO:
        return convert_audio_to_image(inp, out_dir, sr=sr, offset=offset, duration=duration, res_type=res_type,
                                      n_fft=n_fft, hop_length=hop_length, chunk_size=chunk_size, truncate=truncate,
                                      flip=flip, skip=skip)
    elif dtype == DataType.NUMPY:
        return convert_numpy_to_image(inp, out_dir, flip=flip, skip=skip)


def convert_to_audio(inp, out_dir, sr=settings.SAMPLE_RATE, n_fft=settings.N_FFT,
                     hop_length=settings.HOP_LENGTH, fmt=settings.AUDIO_FORMAT,
                     offset=settings.AUDIO_OFFSET, duration=settings.AUDIO_DURATION,
                     flip=settings.IMAGE_FLIP, skip=0):
    dtype = get_data_type(inp, raise_exception=True)
    if dtype == DataType.NUMPY:
        return convert_numpy_to_audio(inp, out_dir, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                      fmt=fmt, offset=offset,
                                      duration=duration, skip=skip)
    elif dtype == DataType.IMAGE:
        return convert_image_to_audio(inp, out_dir, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                      fmt=fmt, offset=offset,
                                      duration=duration, flip=flip, skip=skip)
# endregion
