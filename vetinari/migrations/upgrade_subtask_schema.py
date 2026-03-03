"""Migration: upgrade Subtask schema to include ponder auditing fields.

This script migrates existing Subtask records to include:
- ponder_ranking: []
- ponder_scores: {}
- ponder_used: false
- schema_version: 1 (or higher if extended)

Usage:
    python vetinari/migrations/upgrade_subtask_schema_v1_to_v2.py
"""

import json
import sys
from pathlib import Path


def migrate(storage_root: str = None):
    """Migrate existing subtask files to include ponder fields."""
    if storage_root is None:
        storage_root = Path.home() / ".lmstudio" / "projects" / "Vetinari" / "subtasks"
    else:
        storage_root = Path(storage_root)
    
    if not storage_root.exists():
        print(f"Storage path does not exist: {storage_root}")
        return
    
    migrated_count = 0
    error_count = 0
    
    for file in storage_root.glob("*.json"):
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
                    json.dump(data, f, indent=2)
                print(f"Migrated: {file.name}")
                migrated_count += 1
            else:
                print(f"Skipped (up to date): {file.name}")
                
        except Exception as e:
            print(f"Error migrating {file.name}: {e}")
            error_count += 1
    
    print(f"\nMigration complete:")
    print(f"  Migrated: {migrated_count} files")
    print(f"  Errors: {error_count} files")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        migrate(sys.argv[1])
    else:
        migrate()
