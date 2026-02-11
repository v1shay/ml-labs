# Vishay's ML Labs (Backend Core)

This repository contains the foundational execution engine for Vishay's ML Labs.

Scope (current):
- dataset ingestion (CSV)
- structured dataset profiling
- deterministic, explainable strategy inference (LLM-ready stub)
- orchestration into a clean service layer (FastAPI-mountable later)

Out of scope (current):
- UI
- FastAPI
- training / tuning
- auth
- async

## Structure
- `ml_labs/core/`: core execution engine logic
- `ml_labs/services/`: external integration service abstractions (e.g., GitHub)
- `app/entrypoint.py`: CLI entrypoint for local execution

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the orchestrator on a CSV:

```bash
python app/entrypoint.py /path/to/dataset.csv
```

The output is structured JSON including dataset profile + inferred strategy.
