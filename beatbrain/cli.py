import os
import click
import numpy as np
from tqdm import tqdm
from colorama import Fore

from .defaults import *
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
@click.option('-o', '--out', 'output_dir', required=True)
@click.option('--samplerate', 'sample_rate', default=SAMPLE_RATE, show_default=True)
@click.option('--start', default=AUDIO_START, show_default=True)
@click.option('--duration', default=AUDIO_DURATION, show_default=True)
@click.option('--resample', 'res_type', default=RESAMPLE_TYPE, show_default=True)
@click.option('--fftsize', 'n_fft', default=N_FFT, show_default=True)
@click.option('--hopsize', 'hop_length', default=HOP_LENGTH, show_default=True)
@click.option('--chunksize', 'pixels_per_chunk', default=PIXELS_PER_CHUNK, show_default=True)
@click.option('--truncate/--pad', 'truncate', default=TRUNCATE, show_default=True)
@click.option('--debug/--no-debug', default=False)
def audio_to_images(path, output_dir, sample_rate=SAMPLE_RATE, start=AUDIO_START,
                    duration=AUDIO_DURATION, res_type=RESAMPLE_TYPE,
                    n_fft=N_FFT, hop_length=HOP_LENGTH,
                    pixels_per_chunk=PIXELS_PER_CHUNK, truncate=TRUNCATE, debug=False):
    if os.path.isfile(path):
        files = [path]
        click.echo(f"Converting {Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunk images...")
    elif os.path.isdir(path):
        files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        click.echo(f"Converting {len(files)} files found in "
                   f"{Fore.YELLOW}'{path}'{Fore.RESET} to spectrogram chunk images...")
    else:
        raise click.ClickException(f"{Fore.LIGHTRED_EX}No such file or directory: {Fore.YELLOW}'{path}'{Fore.RESET}")
    # click.confirm(f"This operation will output {len(files)} set(s) of spectrogram chunks to "
    #               f"{Fore.YELLOW}'{output_dir}'{Fore.RESET}. Confirm?\n", abort=True)
    with tqdm(files) as bar:
        for file in bar:
            utils.convert_audio_to_images(file, output_dir, sr=sample_rate, start=start, duration=duration,
                                          res_type=res_type, n_fft=n_fft, hop_length=hop_length,
                                          pixels_per_chunk=pixels_per_chunk, truncate=truncate, debug=debug)
    click.echo(f"\n{Fore.GREEN}Finished converting {len(files)} audio file(s) to spectrogram image chunks.{Fore.RESET}")
    click.echo(f"{Fore.GREEN}The generated spectrogram image chunks are in {Fore.YELLOW}{output_dir}{Fore.RESET}")


@click.command(name='reconstruct', short_help="Reconstruct audio from spectrogram images")
@click.option('-i', '--input', 'path', required=True)
@click.option('-o', '--out', 'output_dir', help='', required=True)
@click.option('-d', '--depth',
              help='How many subdirectories deep to search for images (1 = root only). '
                   'Ignored if input is a single image.', default=WALK_DEPTH)
@click.option('--iter', '--griffinlimiter', 'n_iter', default=GRIFFINLIM_ITER)
@click.option('--hopsize', 'hop_length', default=HOP_LENGTH)
@click.option('--fftsize', 'n_fft', default=N_FFT)
def images_to_audio(path, output_dir, depth=WALK_DEPTH, n_iter=GRIFFINLIM_ITER,
                    hop_length=HOP_LENGTH, n_fft=N_FFT, sr=SAMPLE_RATE, norm=NORMALIZE_AUDIO,
                    fmt=AUDIO_FORMAT, debug=False):
    if os.path.isfile(path):
        click.echo(f"Reconstructing audio from single image: {Fore.YELLOW}'{path}'{Fore.RESET}")
        output = os.path.splitext(os.path.basename(path))[0]
        output = output[:output.rfind('_')]
        output = os.path.join(output_dir, f"{output}.{fmt}")
        recon = utils.convert_images_to_audio([path], output, n_iter=n_iter, hop_length=hop_length,
                                              n_fft=n_fft, sr=sr, norm=norm, fmt=fmt, debug=debug)
    elif os.path.isdir(path):
        directories = utils.walk_dir(path, depth=depth)
        print(directories)
        for directory in directories:
            print(directory)
            files = utils.list_files(directory)
            print(files)
            if files:
                output = os.path.join(output_dir, f"{os.path.basename(directory)}.{fmt}")
                recon = utils.convert_images_to_audio(files, output, n_iter=n_iter, hop_length=hop_length,
                                                      n_fft=n_fft, sr=sr, norm=norm, fmt=fmt, debug=debug)
        click.echo(f"Reconstructing audio from multiple images: {Fore.YELLOW}'{path}'{Fore.RESET}")
