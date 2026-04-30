<div align="center">

<p align="center">
  <img width="900" alt="ML-Labs Architecture" src="https://github.com/user-attachments/assets/20e203c9-9adb-4ae7-ba45-c511b70ac761" />
</p>

<p><strong>An autonomous multi-agent system executing the full machine learning research lifecycle.</strong></p>

</div>

---

## Results

- **Agents:** 19 specialized agents  
- **Research Phases:** 8 end-to-end stages  
- **Pipeline Coverage:** Full lifecycle (data → deployment)  
- **Execution Mode:** Autonomous, multi-agent orchestration  
- **Output:** Validated models + structured research artifacts + APIs  

---

## Overview

ML-Labs is a fully autonomous machine learning research system that replaces fragmented workflows with a unified, agent-driven architecture.

It was built for the Luma AI Agents Hackathon with a demo focused on training an interactive 3D graph neural network with competition datasets.

---

## Research Phases

<div align="center">

| Phase | Description |
|------|------------|
| **1. Initiation & Planning** | Defines objectives, constraints, and system scope |
| **2. Data Foundation** | Ingests, structures, and validates datasets |
| **3. Model Development** | Selects architectures and builds candidate models |
| **4. Validation & Testing** | Runs automated tests and robustness checks |
| **5. Evaluation** | Benchmarks performance and analyzes results |
| **6. Research Output** | Produces structured reports and documentation |
| **7. Execution Layer** | Builds systems, APIs, and deployment pipelines |
| **8. Continuous Loop** | Iterates through feedback and refinement cycles |

</div>


---

## Agent System

<div align="center">

| Agent | Role |
|------|------|
| **Control Agent** | Orchestrates workflow and resource allocation |
| **Problem Framing Agent** | Defines objectives, scope, and success metrics |
| **Data Analysis Agent** | Explores datasets and identifies patterns |
| **Schema Agent** | Designs data structures and transformations |
| **Quality Agent** | Ensures data consistency and reliability |
| **Readiness Agent** | Validates data for modeling |
| **Model Selection Agent** | Chooses algorithms and architectures |
| **Model Agents (Ensemble)** | Builds and iterates candidate models |
| **Optimization Agent** | Performs tuning and search |
| **Testing Agent (Integrated)** | Runs automated validation checks |
| **Field Testing Agent** | Evaluates models in real-world scenarios |
| **Evaluation Agent** | Measures performance vs objectives |
| **Statistical Agent** | Performs statistical analysis and inference |
| **Full Stack Agent** | Designs APIs and infrastructure |
| **SWE Agent** | Implements backend systems |
| **Senior SWE Agent** | Reviews code for scalability and reliability |
| **System Testing Agent** | Executes end-to-end validation |
| **Computing Agent** | Manages environments and execution |
| **Research Agent** | Generates structured outputs and reports |

</div>

---

## Data

- **Sources:** CSV, Kaggle datasets  
- **Type:** structured and semi-structured tabular data  
- **Flow:** ingestion → validation → feature construction → modeling  

Preprocessing:
- schema construction  
- normalization  
- anomaly detection  
- feature engineering  

---

## Experiments / Reproduction

```bash
npm run dev
````

## Run full pipeline:

```bash
curl -X POST http://localhost:3000/api/lab/run
```

## Run inference:

```bash
curl -X POST http://localhost:3000/api/lab/predict
```

Input: raw dataset
Output: trained models + evaluation metrics + predictions

Dependencies

```bash
Node.js
Python 3.x
scikit-learn
NumPy
pandas
```

---

## API

```
GET  /api/lab/demo
POST /api/lab/run
POST /api/lab/predict
```

* `/demo` → deterministic experiment snapshots
* `/run` → full multi-agent execution
* `/predict` → inference on trained models

---

## Repository Structure

```bash
ml-labs/
├── frontend/
├── backend/
├── agents/
├── pipelines/
├── data/
├── experiments/
└── README.md
```

---

## Installation

```bash
npm install

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt
```

---

## Optional

```bash
npm run build
npm start
```
