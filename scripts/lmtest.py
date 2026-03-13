"""Quick LM Studio connection test."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# Load .env
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    for line in open(env_file).readlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if k and v and k not in os.environ:
                os.environ[k] = v

from vetinari.lmstudio_adapter import LMStudioAdapter
from vetinari.model_pool import ModelPool

host = os.environ.get('LM_STUDIO_HOST', 'http://localhost:1234')
token = os.environ.get('LM_STUDIO_API_TOKEN', '')

print(f"Host:       {host}")
print(f"Token set:  {bool(token)}")
print()

# Test 1: Direct GET /v1/models
adapter = LMStudioAdapter(host=host, api_token=token)
result = adapter._get(f'{host}/v1/models')
if result and 'data' in result:
    models = result['data']
    print(f"[PASS] /v1/models: {len(models)} models")
    for m in models[:5]:
        print(f"  - {m.get('id', m)}")
elif result:
    print(f"[WARN] /v1/models returned: {list(result.keys())}")
else:
    print("[FAIL] /v1/models: No response")

print()

# Test 2: ModelPool discovery
pool = ModelPool({'memory_budget_gb': 96}, host, api_token=token, memory_budget_gb=96)
pool.discover_models()
print(f"[{'PASS' if pool.models else 'FAIL'}] ModelPool.discover_models(): {len(pool.models)} models")
for m in pool.models[:3]:
    print(f"  - {m['id']} endpoint={m['endpoint']}")

print()

# Test 3: Chat endpoint
if pool.models:
    model_id = pool.models[0]['id']
    print(f"Testing chat with model: {model_id}")
    result = adapter.chat(model_id, "You are a test assistant.", "Reply with just: OK")
    if result.get('status') == 'ok':
        out = result.get('output', '')[:100]
        print(f"[PASS] chat(): '{out}'")
    else:
        print(f"[FAIL] chat(): {result.get('error', result)}")
else:
    print("[SKIP] No models to test chat with")
