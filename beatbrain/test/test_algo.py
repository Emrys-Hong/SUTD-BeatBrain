import os
import time
import pytest
import librosa
import numpy as np

from beatbrain import utils, settings


def test_stft():
    audio, sr = librosa.load(librosa.util.example_audio_file(), duration=1)
    spec = utils.stft(audio)
    reference = np.abs(librosa.stft(audio))
    reference = reference / reference.max()
    assert np.array_equal(spec, reference)


def test_griffinlim():
    audio, sr = librosa.load(librosa.util.example_audio_file(), duration=1)
    spec = librosa.stft(audio)
    recon = utils.griffinlim(spec, random_state=0)
    reference = librosa.griffinlim(spec, random_state=0)
    assert np.array_equal(recon, reference)


@pytest.mark.parametrize("pixels", [2 ** i for i in range(11)])
@pytest.mark.parametrize("truncate", [True, False])
def test_spec_to_chunks_pad(pixels, truncate):
    chunk_size = 512
    audio, sr = librosa.load(librosa.util.example_audio_file(), duration=1)
    spec = utils.stft(audio)
    utils.spec_to_chunks(spec, chunk_size, False)


@pytest.mark.parametrize("duration", [0.1, 1, 8])
@pytest.mark.parametrize("sr", [32768, 44100, 48000])
@pytest.mark.parametrize("n_fft", [256, 1024,])
def test_audio_to_images(tmp_path, sr, duration, n_fft):
    image_dir = os.path.join(tmp_path, "images")
    utils.convert_audio_to_images(
        librosa.util.example_audio_file(),
        image_dir,
        sr=sr,
        duration=duration,
        n_fft=n_fft,
    )


@pytest.mark.skip("Move image discovery to `utils` before writing this test.")
def test_images_to_audio(tmp_path):
    image_dir = os.path.join(tmp_path, "images")
    recon_dir = os.path.join(tmp_path, "recon")
