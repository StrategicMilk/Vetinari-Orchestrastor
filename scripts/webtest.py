"""Test web UI routes — comprehensive."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

for line in open('.env').readlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        k, v = k.strip(), v.strip()
        if k and v and k not in os.environ:
            os.environ[k] = v

from vetinari.web_ui import app, current_config
print('web_ui loaded OK')
print(f'  host: {current_config["host"]}')
print(f'  token: {"SET" if current_config["api_token"] else "NOT SET"}')
print()

with app.test_client() as client:
    # --- Core routes ---
    r = client.get('/api/status')
    d = r.get_json()
    print(f'/api/status:          {r.status_code}  host={d.get("host")}')

    t0 = time.time()
    r = client.get('/api/models')
    t1 = time.time()
    d = r.get_json()
    models = d.get('models', [])
    print(f'/api/models:          {r.status_code}  count={len(models)}  time={t1-t0:.2f}s  (first — does discovery)')

    t0 = time.time()
    r = client.get('/api/models')
    t1 = time.time()
    d = r.get_json()
    print(f'/api/models (cached): {r.status_code}  count={len(d.get("models",[]))}  time={t1-t0:.3f}s  (should be <0.01s)')

    r = client.get('/api/projects')
    d = r.get_json()
    print(f'/api/projects:        {r.status_code}  count={len(d.get("projects",[]))}')

    # --- Agent routes (serialization fix) ---
    r = client.get('/api/agents/status')
    d = r.get_json()
    if d.get('error'):
        print(f'/api/agents/status:   {r.status_code}  ERROR: {d["error"]}')
    else:
        agents = d.get('agents', [])
        print(f'/api/agents/status:   {r.status_code}  agents={len(agents)}  (empty until initialized = OK)')

    r = client.get('/api/agents/active')
    d = r.get_json()
    if d.get('error'):
        print(f'/api/agents/active:   {r.status_code}  ERROR: {d["error"]}')
    else:
        print(f'/api/agents/active:   {r.status_code}  agents={len(d.get("agents",[]))}')

    r = client.get('/api/agents/tasks')
    d = r.get_json()
    if d.get('error'):
        print(f'/api/agents/tasks:    {r.status_code}  ERROR: {d["error"]}')
    else:
        print(f'/api/agents/tasks:    {r.status_code}  tasks={len(d.get("tasks",[]))}')

    # --- Settings ---
    r = client.get('/api/admin/credentials')
    print(f'/api/admin/creds:     {r.status_code}')

    r = client.get('/api/model-config')
    print(f'/api/model-config:    {r.status_code}')

    # --- Workflow / Tasks ---
    r = client.get('/api/workflow')
    d = r.get_json()
    print(f'/api/workflow:        {r.status_code}  projects={len(d.get("projects",[]))}')

    r = client.get('/api/all-tasks')
    d = r.get_json()
    print(f'/api/all-tasks:       {r.status_code}  tasks={len(d.get("tasks",[]))}')

    # --- Force refresh ---
    r = client.post('/api/models/refresh')
    d = r.get_json()
    print(f'/api/models/refresh:  {r.status_code}  count={d.get("count",0)}')

print()
print('All routes tested.')
