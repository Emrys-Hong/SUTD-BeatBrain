import os
import time
import click
import numpy as np
from tqdm import tqdm
from colorama import Fore
from joblib import Parallel, delayed

from .settings import *
from . import utils


@click.group(invoke_without_command=True)
@click.pass_context
def convert(ctx):
    click.echo(click.style("------------------------\n"
                           "BeatBrain Data Converter\n"
                           "------------------------\n", fg='green', bold=True))
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@click.command(name='spectrogram', short_help="Convert audio to spectrogram images")
@click.option('-i', '--input', 'path', required=True)
@click.option('-o', '--out', 'output_dir', help="Spectrogram output directory", required=True)
@click.option('--samplerate', 'sample_rate', default=SAMPLE_RATE, help="Rate to resample audio at", show_default=True)
@click.option('--start', default=AUDIO_START, help="Audio start offset in seconds", show_default=True)
@click.option('--duration', default=AUDIO_DURATION, type=int, help="Seconds of audio to load", show_default=True)
@click.option('--resample', 'res_type', type=click.Choice(['kaiser_best', 'kaiser_fast']),
              default=RESAMPLE_TYPE, help="Resampling mode", show_default=True)
@click.option('--fftsize', 'n_fft', default=N_FFT, help="FFT window size", show_default=True)
@click.option('--chunksize', 'pixels_per_chunk', default=CHUNK_SIZE,
              help="Horizontal pixels per spectrogram chunk", show_default=True)
@click.option('--skip', default=0, help="Files to skip (Starting from 0)", show_default=True)
@click.option('--truncate/--pad', 'truncate', default=TRUNCATE,
              help="Whether to truncate audio to nearest whole chunk", show_default=True)
@click.option('--debug/--no-debug', default=False)
def audio_to_images(path, output_dir, sample_rate=SAMPLE_RATE, start=AUDIO_START,
                    duration=AUDIO_DURATION, res_type=RESAMPLE_TYPE,
                    n_fft=N_FFT,
                    pixels_per_chunk=CHUNK_SIZE, truncate=TRUNCATE, skip=None, debug=False):
    start_time = time.time()
    path = utils.truepath(path)
    output_dir = utils.truepath(output_dir)
    if os.path.isfile(path):
        files = [path]
        click.echo(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunks...")
    elif os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        if skip:
            click.echo(f"{Fore.MAGENTA}Skipping {skip} / {len(files)} files{Fore.RESET}")
            files = files[skip:]
        click.echo(f"Converting {len(files)} file(s) found in "
                   f"{Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunks...")
    else:
        raise click.ClickException(f"{Fore.LIGHTRED_EX}No such file or directory: {Fore.YELLOW}'{path}'{Fore.RESET}")

    Parallel(n_jobs=NUM_CPUS, verbose=20)(
        delayed(utils.convert_audio_to_images)(
            file, output_dir, sr=sample_rate, start=start, duration=duration,
            res_type=res_type, n_fft=n_fft, pixels_per_chunk=pixels_per_chunk,
            truncate=truncate, debug=debug
        ) for file in files
    )
    click.echo(f"\n{Fore.GREEN}Converted {len(files)} audio file(s) to spectrograms "
               f"in {time.time() - start_time:.2f}s.{Fore.RESET}")
    click.echo(f"\n{Fore.GREEN}The generated spectrogram chunks are in {Fore.YELLOW}{output_dir}{Fore.RESET}\n")


@click.command(name='reconstruct', short_help="Reconstruct audio from spectrogram images")
@click.option('-i', '--input', 'path', required=True)
@click.option('-o', '--out', 'output_dir', help='', required=True)
@click.option('-d', '--depth',
              help='How many subdirectories deep to search for images (1 = root only). '
                   'Ignored if input is a single image.', default=WALK_DEPTH)
@click.option('--iter', '--griffinlimiter', 'n_iter', default=GRIFFINLIM_ITER)
@click.option('--fftsize', 'n_fft', default=N_FFT)
@click.option('--samplerate', 'sample_rate', default=SAMPLE_RATE)
@click.option('--norm', 'norm', default=NORMALIZE_AUDIO)
@click.option('-f', '--format', 'fmt', default=AUDIO_FORMAT)
@click.option('--skip', default=0, help="Directories to skip (Starting from 0)", show_default=True)
def images_to_audio(path, output_dir, depth=WALK_DEPTH, n_iter=GRIFFINLIM_ITER,
                    n_fft=N_FFT, sample_rate=SAMPLE_RATE, norm=NORMALIZE_AUDIO,
                    fmt=AUDIO_FORMAT, skip=None, debug=False):
    start_time = time.time()
    path = utils.truepath(path)
    output_dir = utils.truepath(output_dir)
    num_tracks = 0
    if os.path.isfile(path):
        num_tracks = 1
        click.echo(f"Reconstructing audio from single image: {Fore.YELLOW}'{path}'{Fore.RESET}")
        output = os.path.splitext(os.path.basename(path))[0]
        output = output[:output.rfind('_')]
        output = os.path.join(output_dir, f"{output}.{fmt}")
        utils.convert_images_to_audio([path], output, n_iter=n_iter, n_fft=n_fft,
                                      sr=sample_rate, norm=norm, fmt=fmt, debug=debug)
    elif os.path.isdir(path):
        directories = [d for d in utils.walk_dir(path, depth=depth) if utils.list_files(d)]
        if skip:
            click.echo(f"{Fore.MAGENTA}Skipping {skip} / {len(directories)} directories{Fore.RESET}")
            directories = directories[skip:]
        num_tracks = len(directories)
        if num_tracks == 0:
            raise click.ClickException(f"{Fore.LIGHTRED_EX}Could not find any (valid) images to"
                                       f" reconstruct audio from under {Fore.YELLOW}'{path}'{Fore.RESET}")
        click.echo(f"Reconstructing {num_tracks} audio track(s) from {Fore.YELLOW}'{path}'{Fore.RESET}")

        jobs = {directory: utils.list_files(directory) for directory in directories}
        Parallel(n_jobs=NUM_CPUS, verbose=20)(
            delayed(utils.convert_images_to_audio)(
                files, os.path.join(output_dir, f"{os.path.basename(directory)}.{fmt}"),
                n_iter=n_iter, n_fft=n_fft, sr=sample_rate,
                norm=norm, fmt=fmt, debug=debug
            ) for directory, files in jobs.items()
        )

    click.echo(f"\n{Fore.GREEN}Converted {num_tracks} spectrograms "
               f"to audio in {time.time() - start_time:.2f}s{Fore.RESET}\n")
    click.echo(f"\n{Fore.GREEN}The generated audio tracks are in {Fore.YELLOW}{output_dir}{Fore.RESET}\n")
