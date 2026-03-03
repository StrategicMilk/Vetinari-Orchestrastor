import os
import sys
import json
import yaml
import argparse
from pathlib import Path
import logging

from vetinari.orchestrator import Orchestrator
from vetinari.utils import setup_logging, load_config

def main():
    parser = argparse.ArgumentParser(description="Vetinari: Local LLM Orchestration")
    parser.add_argument("--config", default="manifest/vetinari.yaml",
                       help="Path to the manifest file")
    parser.add_argument("--task", help="Run a specific task by ID")
    parser.add_argument("--upgrade", action="store_true",
                       help="Check for and install model upgrades")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found")
        sys.exit(1)
    
    config = load_config(config_path)
    
    orchestrator = Orchestrator(str(config_path))
    
    if args.upgrade:
        orchestrator.check_and_upgrade_models()
    elif args.task:
        orchestrator.run_task(args.task)
    else:
        orchestrator.run_all()

if __name__ == "__main__":
    main()