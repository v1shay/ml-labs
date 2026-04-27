# ML-Labs

**An entire research lab at your fingertips.**

---

ML-Labs is an end-to-end machine learning platform where autonomous agents replicate the workflow of a full research team.

Instead of manually assembling pipelines, sourcing data, and iterating on models, ML-Labs coordinates a system of agents that:

* Discover and ingest datasets (CSV or Kaggle)
* Profile and analyze data distributions
* Construct and execute experiments
* Train, test, and optimize models
* Run statistical validation and accuracy analysis
* Generate visualizations and diagnostics
* Produce structured reports and research-style outputs
* Serve predictions through a unified API

From raw data to validated results and documentation—fully automated.

---

## Agent System

ML-Labs is organized as a coordinated swarm of specialized agents, each responsible for a stage of the research lifecycle:

* **Data Sourcing Agent**
  Locates and ingests datasets (local or Kaggle)

* **Data Analyst Agent**
  Profiles distributions, detects anomalies, and prepares features

* **Computation Agent**
  Grounds research in real math and physics in a user-facing manner through KaTex

* **Experiment Engineer Agent**
  Designs experiment structure and selects modeling strategies

* **Modeling Agent**
  Trains and evaluates multiple sklearn models across tasks

* **Optimization Agent**
  Iterates on configurations to improve performance

* **Statistical Analysis Agent**
  Runs validation, accuracy metrics, and comparative evaluation

* **Visualization Agent**
  Generates plots, graphs, and interpretable outputs

* **Report / Research Agent**
  Produces structured summaries, experiment logs, and research-style drafts

* **Prediction Agent**
  Exposes trained models for inference via API

These agents operate as a coordinated system, not isolated tools.

---

## API Surface

* `GET /api/lab/demo`
  Deterministic, real-backed experiment snapshots

* `POST /api/lab/run`
  Executes full multi-agent experiment pipeline

* `POST /api/lab/predict`
  Serves predictions from completed runs

---

## Tech Stack

* **Backend:** Node.js (API orchestration layer)
* **ML Execution:** Python (scikit-learn, pandas, NumPy)
* **Experiment Engine:** Custom multi-agent orchestration system
* **Data Sources:** CSV + Kaggle datasets
* **Frontend:** Next.js (consumer layer for outputs and visualization)

---

## Run locally

```bash id="ml1"
npm install

python3 -m venv .venv
. .venv/bin/activate

pip install -r requirements.txt

npm run dev
```

---

## Example

Run a full experiment:

```bash id="ml2"
curl -X POST http://localhost:3000/api/lab/run \
-F "file=@data.csv" \
-F "targetColumn=target" \
-F "intentPrompt=Predict target variable"
```

---

## Why it matters

Machine learning today is constrained by coordination cost.

A single experiment requires:

* Data acquisition
* Cleaning and profiling
* Model design
* Iteration and tuning
* Evaluation and reporting

ML-Labs collapses this into a single execution layer.

Not by simplifying the work—but by automating the entire research loop.

---

ML-Labs turns machine learning from a manual process into an autonomous system.
