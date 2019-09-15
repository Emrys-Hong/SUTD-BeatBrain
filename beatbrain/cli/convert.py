import os
import time
import click
import numpy as np
from tqdm import tqdm
from colorama import Fore
from joblib import Parallel, delayed

from .. import settings, utils


@click.group(invoke_without_command=True, short_help="Data Conversion Utilities")
@click.pass_context
def convert(ctx):
    click.echo(click.style("------------------------\n"
                           "BeatBrain Data Converter\n"
                           "------------------------\n", fg='green', bold=True))
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@convert.command(name='spectrogram', short_help="Convert audio file(s) to spectrogram array(s)")
@click.option('-i', '--input', 'path', help="Path to audio file(s)", required=True)
@click.option('-o', '--output', 'output', help="Spectrogram output directory", required=True)
@click.option('--sr', help="Rate at which to resample audio", default=settings.SAMPLE_RATE, show_default=True)
@click.option('--offset', help="Audio start timestamp (seconds)", default=settings.AUDIO_START, show_default=True)
@click.option('--duration', help="Audio duration (seconds)", default=settings.AUDIO_DURATION, show_default=True)
@click.option('--n_fft', help="Size of FFT window to use", default=settings.N_FFT, show_default=True)
@click.option('--hop_length', help="Short-time Fourier Transform hop length", default=settings.HOP_LENGTH,
              show_default=True)
@click.option('--n_mels', help="Number of frequency bins to use", default=settings.N_MELS, show_default=True)
@click.option('--chunk_size', help="Number of frames per spectrogram chunk", default=settings.CHUNK_SIZE,
              show_default=True)
@click.option('--skip', help="Number of samples to skip. Useful when restarting a failed job.", default=0,
              show_default=True)
def to_tfrecords(path, output, **kwargs):
    return utils.convert_audio_to_arrays(path, output, **kwargs)


@convert.command(name='spectrogram_legacy', short_help="Convert audio to spectrogram images")
@click.option('-i', '--input', 'path', required=True)
@click.option('-o', '--out', 'output_dir', help="Spectrogram output directory", required=True)
@click.option('--samplerate', 'sample_rate', default=settings.SAMPLE_RATE, help="Rate to resample audio at",
              show_default=True)
@click.option('--start', default=settings.AUDIO_START, help="Audio start offset in seconds", show_default=True)
@click.option('--duration', default=settings.AUDIO_DURATION, type=int, help="Seconds of audio to load",
              show_default=True)
@click.option('--resample', 'res_type', type=click.Choice(['kaiser_best', 'kaiser_fast']),
              default=settings.RESAMPLE_TYPE, help="Resampling mode", show_default=True)
@click.option('--fftsize', 'n_fft', default=settings.N_FFT, help="FFT window size", show_default=True)
@click.option('--chunksize', 'pixels_per_chunk', default=settings.CHUNK_SIZE,
              help="Horizontal pixels per spectrogram chunk", show_default=True)
@click.option('--skip', default=0, help="Files to skip (Starting from 0)", show_default=True)
@click.option('--truncate/--pad', 'truncate', default=settings.TRUNCATE,
              help="Whether to truncate audio to nearest whole chunk", show_default=True)
@click.option('--debug/--no-debug', default=False)
def audio_to_images(path, output_dir, sample_rate=settings.SAMPLE_RATE, start=settings.AUDIO_START,
                    duration=settings.AUDIO_DURATION, res_type=settings.RESAMPLE_TYPE,
                    n_fft=settings.N_FFT,
                    pixels_per_chunk=settings.CHUNK_SIZE, truncate=settings.TRUNCATE, skip=None, debug=False):
    start_time = time.time()
    path = utils.truepath(path)
    output_dir = utils.truepath(output_dir)
    if settings.os.path.isfile(path):
        files = [path]
        click.echo(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunks...")
    elif settings.os.path.isdir(path):
        files = [settings.os.path.join(path, file) for file in settings.os.listdir(path) if
                 settings.os.path.isfile(settings.os.path.join(path, file))]
        if skip:
            click.echo(f"{Fore.MAGENTA}Skipping {skip} / {len(files)} files{Fore.RESET}")
            files = files[skip:]
        click.echo(f"Converting {len(files)} file(s) found in "
                   f"{Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunks...")
    else:
        raise click.ClickException(f"{Fore.LIGHTRED_EX}No such file or directory: {Fore.YELLOW}'{path}'{Fore.RESET}")

    Parallel(n_jobs=settings.NUM_CPUS, verbose=20)(
        delayed(utils.convert_audio_to_images)(
            file, output_dir, sr=sample_rate, start=start, duration=duration,
            res_type=res_type, n_fft=n_fft, pixels_per_chunk=pixels_per_chunk,
            truncate=truncate, debug=debug
        ) for file in files
    )
    click.echo(f"\n{Fore.GREEN}Converted {len(files)} audio file(s) to spectrograms "
               f"in {time.time() - start_time:.2f}s.{Fore.RESET}")
    click.echo(f"\n{Fore.GREEN}The generated spectrogram chunks are in {Fore.YELLOW}{output_dir}{Fore.RESET}\n")


@convert.command(name='reconstruct', short_help="Reconstruct audio from spectrogram images")
@click.option('-i', '--input', 'path', required=True)
@click.option('-o', '--out', 'output_dir', help='', required=True)
@click.option('-d', '--depth',
              help='How many subdirectories deep to search for images (1 = root only). '
                   'Ignored if input is a single image.', default=settings.WALK_DEPTH)
@click.option('--iter', '--griffinlimiter', 'n_iter', default=settings.GRIFFINLIM_ITER)
@click.option('--fftsize', 'n_fft', default=settings.N_FFT)
@click.option('--samplerate', 'sample_rate', default=settings.SAMPLE_RATE)
@click.option('--norm', 'norm', default=settings.NORMALIZE_AUDIO)
@click.option('-f', '--format', 'fmt', default=settings.AUDIO_FORMAT)
@click.option('--skip', default=0, help="Directories to skip (Starting from 0)", show_default=True)
def images_to_audio(path, output_dir, depth=settings.WALK_DEPTH, n_iter=settings.GRIFFINLIM_ITER,
                    n_fft=settings.N_FFT, sample_rate=settings.SAMPLE_RATE, norm=settings.NORMALIZE_AUDIO,
                    fmt=settings.AUDIO_FORMAT, skip=None, debug=False):
    start_time = time.time()
    path = utils.truepath(path)
    output_dir = utils.truepath(output_dir)
    num_tracks = 0
    if settings.os.path.isfile(path):
        num_tracks = 1
        click.echo(f"Reconstructing audio from single image: {Fore.YELLOW}'{path}'{Fore.RESET}")
        output = settings.os.path.splitext(settings.os.path.basename(path))[0]
        output = output[:output.rfind('_')]
        output = settings.os.path.join(output_dir, f"{output}.{fmt}")
        utils.convert_images_to_audio([path], output, n_iter=n_iter, n_fft=n_fft,
                                      sr=sample_rate, norm=norm, fmt=fmt, debug=debug)
    elif settings.os.path.isdir(path):
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
        Parallel(n_jobs=settings.NUM_CPUS, verbose=20)(
            delayed(utils.convert_images_to_audio)(
                files, settings.os.path.join(output_dir, f"{settings.os.path.basename(directory)}.{fmt}"),
                n_iter=n_iter, n_fft=n_fft, sr=sample_rate,
                norm=norm, fmt=fmt, debug=debug
            ) for directory, files in jobs.items()
        )

    click.echo(f"\n{Fore.GREEN}Converted {num_tracks} spectrograms "
               f"to audio in {time.time() - start_time:.2f}s{Fore.RESET}\n")
    click.echo(f"\n{Fore.GREEN}The generated audio tracks are in {Fore.YELLOW}{output_dir}{Fore.RESET}\n")
