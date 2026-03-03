# Vetinari Orchestrator

## Overview
Vetinari is a local LLM orchestration system that automatically assigns tasks to the most suitable models, schedules execution, validates outputs, and builds runnable artifacts.

## Quick Start
1. Ensure Python 3.9+ is installed
2. Install dependencies: `pip install requests pyyaml`
3. Run the orchestrator: `python cli.py`
4. To run specific tasks: `python cli.py --task t1`
5. To check for upgrades: `python cli.py --upgrade`

## Project Structure
```
Vetinari/
├── cli.py                 # Main CLI entry point
├── orchestrator.py       # Core orchestration logic
├── lmstudio_adapter.py   # LM Studio REST API wrapper
├── model_pool.py         # Model discovery and scoring
├── scheduler.py          # Task dependency scheduling
├── executor.py           # Task execution engine
├── validator.py          # Output validation
├── builder.py            # Artifact compilation
├── upgrader.py           # Model upgrade management
├── utils.py              # Utility functions
├── manifest/
│   └── vetinari.yaml     # Project configuration
├── models/              # Model metadata
├── prompts/             # Task prompts
├── tasks/               # Task specifications
├── outputs/             # Task outputs
├── benchmarks/          # Performance data
├── build/               # Build artifacts
└── logs/                # Runtime logs
```

## Configuration
Edit `manifest/vetinari.yaml` to configure:
- Model endpoints and capabilities
- Task definitions and dependencies
- Upgrade policies
- Output strategies

## Architecture
1. **Auto-discovery**: Scans LM Studio registry for available models (memory <= 96GB)
2. **Scoring**: Assigns tasks to best model based on capability, latency, reliability, context fit
3. **Scheduling**: Builds DAG from dependencies, executes tasks in parallel where possible
4. **Execution**: Prompts assigned model via LM Studio API, captures outputs
5. **Validation**: Checks syntax and basic correctness of generated artifacts
6. **Building**: Assembles validated outputs into runnable Python CLI scaffold
7. **Upgrades**: Periodically checks for better models, requires approval before installation

## Requirements
- Python 3.9+
- LM Studio running at http://10.0.0.96:1234
- Windows 10/11 (64-bit)
- 64GB RAM, RTX 5090 GPU recommended

## Usage Examples
```bash
# Run full workflow
python cli.py

# Run specific task
python cli.py --task t1

# Check for model upgrades
python cli.py --upgrade

# Verbose logging
python cli.py --verbose
```

## License
MIT License - see LICENSE file for details.