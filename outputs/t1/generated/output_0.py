import os
import sys
from typing import List

# Task list with dependencies:
tasks = [
    {"name": "Create CLI Scaffold", "depends_on": ["get_default_config"]},
    {"name": "Create API Scaffold", "depends_on": []}
]

def get_default_config():
    # This function would contain the default configuration for the application.
    # For now, it's a placeholder.
    return {}

def create_cli_scaffold():
    """Creates a basic CLI scaffold."""
    cli = click Commander()
    
    @cli.command()
    def hello():
        """A simple 'hello' command."""
        print("Hello from the CLI!")

    cli.add_command(hello)

    cli.add_argument("-n", "--name", help="The name of the application.")
    cli.add_argument("--version", action="store_true", help="Show version.")

    return cli

def create_api_scaffold():
    """Creates a basic API scaffold using FastAPI."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class User(BaseModel):
        name: str
        email: str

    @app.post("/users", response_model=User)
    def create_user(user: User):
        return user

    @app.get("/users/{user_id}")
    def get_user(user_id: int):
        return {"name": "John Doe", "email": "john.doe@example.com"}

if __name__ == "__main__":
    # Recommended directory layout:
    cli_scaffold_dir = os.path.join("cli", "scaffold")
    api_scaffold_dir = os.path.join("api", "scaffold")

    os.makedirs(cli_scaffold_dir, exist_ok=True)
    os.makedirs(api_scaffold_dir, exist_ok=True)

    tasks["create_cli_scaffold"].run(cli_scaffold_dir)
    tasks["create_api_scaffold"].run(api_scaffold_dir)