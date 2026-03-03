import click

@click.group()
def cli():
    pass

@cli.command()
@click.version_option(version="1.0.0", message="Vetinari API CLI")
def version():
    """Prints the version number of the application."""
    click.echo(__version__)

if __name__ == '__main__':
    cli()