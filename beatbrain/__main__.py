import click

from .defaults import *
from . import cli


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    click.echo(
        click.style("BeatBrain", fg='bright_cyan', bold=True, underline=True)
    )
    click.echo(
        click.style("=========================================================================\n"
                    "BeatBrain (©2019, Krishna Penukonda) is distributed under the MIT License\n"
                    "=========================================================================\n", fg='cyan')
    )
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


main.add_command(cli.audio_to_images)
main.add_command(cli.images_to_audio)

if __name__ == '__main__':
    main()
