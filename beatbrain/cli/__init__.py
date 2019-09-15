import click

from . import convert


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


main.add_command(convert.convert)
