<p align="center">
  <img width="900" alt="ML-Labs Screenshot" src="https://github.com/user-attachments/assets/20e203c9-9adb-4ae7-ba45-c511b70ac761" />
</p>

<h2 align="center">An entire research lab at your fingertips.</h2>

<p align="center">
ML-Labs is an autonomous machine learning system that executes the full research lifecycle end-to-end.
</p>

---

## Overview

ML-Labs replaces fragmented ML workflows with a coordinated multi-agent system.

Instead of manually assembling pipelines, sourcing data, and iterating on models, ML-Labs runs the entire loop:

- Data discovery and ingestion  
- Statistical profiling and feature preparation  
- Experiment design and execution  
- Model training, testing, and optimization  
- Validation and comparative analysis  
- Visualization and diagnostics  
- Structured research outputs  
- Production-ready prediction APIs  

From raw data to validated results—fully automated.

---

## Agent System

ML-Labs operates as a tightly coordinated system of specialized agents:

- **Data Sourcing Agent**  
  Ingests datasets (CSV, Kaggle)

- **Data Analyst Agent**  
  Profiles distributions, detects anomalies, prepares features

- **Computation Agent**  
  Grounds outputs in formal math (KaTeX-rendered)

- **Experiment Engineer Agent**  
  Designs experiments and selects modeling strategies

- **Modeling Agent**  
  Trains and evaluates multiple models (scikit-learn)

- **Optimization Agent**  
  Iterates configurations for performance gains

- **Statistical Analysis Agent**  
  Runs validation, metrics, and comparative evaluation

- **Visualization Agent**  
  Produces interpretable plots and diagnostics

- **Research Agent**  
  Generates structured reports and experiment logs

- **Prediction Agent**  
  Serves trained models via API

Agents operate as a unified system—not isolated tools.

---

## API

GET  /api/lab/demo
POST /api/lab/run
POST /api/lab/predict

- `/demo` → deterministic experiment snapshots  
- `/run` → full multi-agent pipeline execution  
- `/predict` → inference on trained models  

---

## Tech Stack

- **Backend:** Node.js (orchestration layer)  
- **ML Execution:** Python (scikit-learn, pandas, NumPy)  
- **Engine:** Custom multi-agent system  
- **Data Sources:** CSV, Kaggle  
- **Frontend:** Next.js

---

## Local Setup

```bash
npm install

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt

npm run dev
# ml-labs
