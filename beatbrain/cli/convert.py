import os
import time
import click
import numpy as np
from tqdm import tqdm
from colorama import Fore
# from joblib import Parallel, delayed

from .. import settings, utils


@click.group(invoke_without_command=True, short_help="Data Conversion Utilities")
@click.pass_context
def convert(ctx):
    click.echo(click.style("------------------------\n"
                           "BeatBrain Data Converter\n"
                           "------------------------\n", fg='green', bold=True))
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@convert.command(name='numpy', short_help="Convert audio files or TIFF images to numpy arrays")
@click.argument('path')
@click.argument('output')
@click.option('--sr', help="Rate at which to resample audio", default=settings.SAMPLE_RATE, show_default=True)
@click.option('--offset', help="Audio start timestamp (seconds)", default=settings.AUDIO_OFFSET, show_default=True)
@click.option('--duration', help="Audio duration (seconds)", default=settings.AUDIO_DURATION,
              type=float, show_default=True)
@click.option('--n_fft', help="Size of FFT window to use", default=settings.N_FFT, show_default=True)
@click.option('--hop_length', help="Short-time Fourier Transform hop length", default=settings.HOP_LENGTH,
              show_default=True)
@click.option('--n_mels', help="Number of frequency bins to use", default=settings.N_MELS, show_default=True)
@click.option('--chunk_size', help="Number of frames per spectrogram chunk", default=settings.CHUNK_SIZE,
              show_default=True)
@click.option('--flip', help="Whether to flip images veritcally", default=settings.IMAGE_FLIP, show_default=True)
@click.option('--truncate/--pad', help="Whether to truncate or pad the last chunk", default=True,
              show_default=True)
@click.option('--skip', help="Number of samples to skip. Useful when restarting a failed job.", default=0,
              show_default=True)
def to_numpy(path, output, **kwargs):
    return utils.convert_to_numpy(path, output, **kwargs)


@convert.command(name='image', short_help="Convert audio or .npz files to TIFF images")
@click.argument('path')
@click.argument('output')
@click.option('--sr', help="Rate at which to resample audio", default=settings.SAMPLE_RATE, show_default=True)
@click.option('--offset', help="Audio start timestamp (seconds)", default=settings.AUDIO_OFFSET, show_default=True)
@click.option('--duration', help="Audio duration (seconds)", default=settings.AUDIO_DURATION,
              type=float, show_default=True)
@click.option('--n_fft', help="Size of FFT window to use", default=settings.N_FFT, show_default=True)
@click.option('--hop_length', help="Short-time Fourier Transform hop length", default=settings.HOP_LENGTH,
              show_default=True)
@click.option('--chunk_size', help="Number of frames per spectrogram chunk", default=settings.CHUNK_SIZE,
              show_default=True)
@click.option('--truncate/--pad', help="Whether to truncate or pad the last chunk", default=True,
              show_default=True)
@click.option('--flip', help="Whether to flip images veritcally", default=settings.IMAGE_FLIP, show_default=True)
@click.option('--skip', help="Number of data samples to skip. Useful when restarting a failed job.", default=0,
              show_default=True)
def to_image(path, output, **kwargs):
    return utils.convert_to_image(path, output, **kwargs)


@convert.command(name='audio', short_help="Convert .npz files or TIFF images to audio files")
@click.argument('path')
@click.argument('output')
@click.option('--sr', help="Rate at which to resample audio", default=settings.SAMPLE_RATE, show_default=True)
@click.option('--n_fft', help="Size of FFT window to use", default=settings.N_FFT, show_default=True)
@click.option('--hop_length', help="Short-time Fourier Transform hop length", default=settings.HOP_LENGTH,
              show_default=True)
@click.option('--offset', help="Start point (in seconds) of reconstructed audio", default=settings.AUDIO_OFFSET,
              show_default=True)
@click.option('--duration', help="Maximum seconds of audio to convert", default=settings.AUDIO_DURATION,
              type=float, show_default=True)
@click.option('--flip', help="Whether to flip images veritcally", default=settings.IMAGE_FLIP, show_default=True)
@click.option('--skip', help="Number of samples to skip. Useful when restarting a failed job.", default=0,
              show_default=True)
def to_audio(path, output, **kwargs):
    return utils.convert_to_audio(path, output, **kwargs)
