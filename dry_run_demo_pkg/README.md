# Dry Run Demo Package

This is a minimal Python package scaffold for demonstrating Vetinari's end-to-end
Plan -> Approve -> Execute -> Verify workflow.

## Purpose

This package serves as a toy task for testing Vetinari's:

- Dual memory backends (OcMemoryStore + MnemosyneMemoryStore)
- Plan gating across all domains
- Approval workflow with audit logging
- Coding bridge scaffold generation

## Usage

```bash
# Install the package
cd dry_run_demo_pkg
pip install -e .

# Run the demo
python -m dry_run_demo_pkg
```

## Structure

```
dry_run_demo_pkg/
├── setup.py              # Package setup configuration
├── README.md             # This file
└── dry_run_demo_pkg/
    ├── __init__.py      # Package initialization
    └── __main__.py      # CLI entry point
```

## Vetinari Flow

1. **Plan**: Generate plan to scaffold this package
2. **Approve**: Human approval via plan API (or auto-approve for low-risk)
3. **Execute**: CodingBridge generates scaffold files
4. **Verify**: Confirm files exist and log to dual memory
