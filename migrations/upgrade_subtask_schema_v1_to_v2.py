"""Migration: upgrade Subtask schema to include ponder auditing fields.

This script migrates existing Subtask records to include:
- ponder_ranking: []
- ponder_scores: {}
- ponder_used: false
- schema_version: 1 (or higher if extended)
"""

import json
import pathlib
from pathlib import Path
from vetinari.subtask_tree import Subtask
from vetinari.subtask_tree import SubtaskTree
from vetinari.config import get_subdirectory

def migrate(storage_root: str = None):
    root = Path(storage_root) if storage_root else get_subdirectory("subtasks")
    # For each plan JSON in the subtasks folder, ensure ponder fields exist
    for file in root.glob("*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            subtasks = data.get('subtasks', [])
            changed = False
            for st in subtasks:
                if 'ponder_ranking' not in st:
                    st['ponder_ranking'] = []
                    changed = True
                if 'ponder_scores' not in st:
                    st['ponder_scores'] = {}
                    changed = True
                if 'ponder_used' not in st:
                    st['ponder_used'] = False
                    changed = True
                if 'schema_version' not in st:
                    st['schema_version'] = 1
                    changed = True
            if changed:
                with open(file, 'w') as f:
                    json.dump({'plan_id': file.stem, 'subtasks': subtasks}, f, indent=2)
                print(f"Migrated {file}")
        except Exception as e:
            print(f"Migration failed for {file}: {e}")

if __name__ == '__main__':
    migrate()
