# 🚀 LLM Evaluation Harness

> **Production-grade framework for benchmarking Large Language Models (LLMs) for reliability, consistency, and phrasing robustness.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue)
![Tests](https://img.shields.io/badge/Tests-34%20Passing-success)

---

## 📖 Overview

LLMs often produce different answers to semantically identical questions that are phrased differently. This project measures that behavior using a reproducible evaluation pipeline.

### Highlights

- ✅ 25 benchmark questions
- ✅ 75 prompts (3 phrasings each)
- ✅ Exact Match scoring
- ✅ Numeric scoring
- ✅ LLM-as-a-Judge
- ✅ JSON & Markdown reports
- ✅ Docker support
- ✅ Modular architecture
- ✅ Pytest test suite

---

# 🏗 Architecture

```text
Dataset
   │
   ▼
Prompt Generator
   │
   ▼
Target LLM
   │
   ▼
Response Collector
   │
   ▼
Scoring Engine
 ├── Exact Match
 ├── Numeric Match
 └── LLM Judge
   │
   ▼
Analyzer
   │
   ▼
Reports
```

---

# ✨ Features

- Evaluate local Ollama models
- Compare consistency across prompt variations
- Multiple scoring strategies
- Category and difficulty analytics
- Phrasing sensitivity metric
- Dockerized execution
- Easily extensible dataset

---

# 📂 Project Structure

```text
llm-eval-harness/
├── data/
│   └── dataset.json
├── results/
│   ├── raw_responses.json
│   ├── scored_results.json
│   ├── report.json
│   └── summary_report.md
├── src/
│   ├── main.py
│   ├── runner.py
│   ├── scorer.py
│   └── analyzer.py
├── tests/
│   ├── test_scorer.py
│   ├── test_analyzer.py
│   └── test_providers.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🧠 What this project does

LLM Evaluation Harness benchmarks how reliably an LLM answers the same question when it is phrased differently. It runs a curated benchmark dataset, collects responses from a target model, scores them with exact match, numeric matching, or LLM-as-judge logic, and generates JSON and Markdown reports.

Each question has exactly three phrasings, so the project can measure consistency and prompt sensitivity in a repeatable way.

---

## 🚀 Quick start (free local option with Ollama)

This is the recommended path if you do not want to use paid cloud APIs.

### 1. Install and start Ollama

```bash
ollama serve
```

### 2. Pull a small local model

```bash
ollama pull llama3.2:3b
```

### 3. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 4. Configure the environment

Create a `.env` file with:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
TARGET_MODEL=llama3.2:3b
JUDGE_MODEL=llama3.2:3b
MAX_CALLS=3
```

### 5. Run the evaluation

```bash
python -m src.main --provider ollama --max-calls 3
```

This runs the full pipeline:
1. Query the model
2. Score the responses
3. Generate reports in the `results/` folder

---

## ☁️ Provider setup

This project uses Ollama only. No cloud API keys are required.

Use the following environment values:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
```

---

## 🧪 Testing

```bash
pytest -q tests/test_providers.py tests/test_scorer.py tests/test_analyzer.py
```

The suite currently covers 34 tests.

---

## 📄 Output files

After a successful run, these files are written to the `results/` directory:

- `raw_responses.json`
- `scored_results.json`
- `report.json`
- `summary_report.md`

---

## 🔧 Notes

- The project is built to highlight prompt sensitivity and inconsistent answers across phrasing variations.
- The free local Ollama setup avoids API quota issues completely.
- If you want, you can extend the benchmark dataset in `data/dataset.json` to test additional questions.
