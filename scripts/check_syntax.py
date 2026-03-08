"""Syntax check all Python files in the Vetinari project."""
import os
import ast

base = os.path.dirname(os.path.abspath(__file__))
errors = []
checked = 0
# Skip generated output directories and test artifacts
skip_dirs = {
    'venv', '.git', '__pycache__', 'vetinari.egg-info', 'build', 'dist',
    '.pytest_cache', 'outputs', 'projects',  # Skip generated artifacts
}

for root, dirs, files in os.walk(base):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        rel = os.path.relpath(path, base)
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                src = fh.read()
            ast.parse(src)
            checked += 1
        except SyntaxError as e:
            errors.append((rel, str(e).encode('ascii', 'replace').decode()))
        except Exception:
            pass

print(f'Checked {checked} Python source files (excl. outputs/)')
if errors:
    print(f'\nFOUND {len(errors)} SYNTAX ERROR(S):')
    for rel, e in errors:
        print(f'  {rel}: {e}')
else:
    print('All source files OK -- no syntax errors found!')
