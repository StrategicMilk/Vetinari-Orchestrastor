# Vetinari — Quick Start Guide

Get from zero to a running system in under 10 minutes.

---

## Prerequisites

| Requirement | Minimum version | Check |
|---|---|---|
| Python | 3.9 | `python --version` |
| pip | 22+ | `pip --version` |
| LM Studio | 0.3+ | Running locally |
| Git | any | `git --version` |

---

## 1. Clone and install

```bash
git clone https://github.com/your-org/vetinari.git
cd vetinari
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 2. Configure LM Studio connection

Copy the example config and edit the host:

```bash
cp .env.example .env
```

Open `.env` and set:

```
LMSTUDIO_HOST=http://localhost:1234   # or your LM Studio IP
```

---

## 3. Verify the installation

```bash
python -c "import vetinari; print('Vetinari OK')"
pytest tests/ -q --tb=short
```

All tests should pass. See `TEST_REPORT.md` for the expected baseline.

---

## 4. Start the monitoring dashboard

```bash
python examples/dashboard_rest_api_example.py
```

Open `http://localhost:5000/dashboard` in your browser. You should see the
real-time monitoring UI.

---

## 5. Run your first task

```python
from vetinari.telemetry import get_telemetry_collector
from vetinari.dashboard.api import get_dashboard_api

# Simulate a model call
tel = get_telemetry_collector()
tel.record_adapter_latency("lmstudio", "llama-3", 120.5, success=True, tokens_used=300)

# Check the dashboard API
api = get_dashboard_api()
snap = api.get_latest_metrics()
print(snap.to_dict()["adapters"])
```

---

## 6. Key entry points

| What | Where |
|---|---|
| CLI | `python -m vetinari.cli --help` |
| Web UI | `python vetinari/web_ui.py` → `http://localhost:5000` |
| Monitoring dashboard | `python examples/dashboard_rest_api_example.py` |
| Analytics example | `python examples/analytics_example.py` |
| Dashboard Python API | `python examples/dashboard_example.py` |

---

## Next steps

- Read `docs/getting-started/onboarding.md` for a full onboarding guide
- Read `.claude/docs/architecture.md` for architecture and conventions
- See `docs/reference/production.md` before deploying to production
- Browse `examples/` for more usage patterns
