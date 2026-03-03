import logging
from pathlib import Path
import yaml


def setup_logging(level=logging.INFO, log_dir="logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=level,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(f"{log_dir}/vetinari.log", mode="a"),
                                  logging.StreamHandler()])


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str):
    return load_yaml(path)
