# 🚀 LLM Evaluation Harness

> **Production-grade framework for benchmarking Large Language Models (LLMs) for reliability, consistency, and phrasing robustness.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![Claude](https://img.shields.io/badge/Anthropic-Claude-purple)
![Tests](https://img.shields.io/badge/Tests-32%20Passing-success)

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

- Evaluate Gemini and Claude models
- Compare consistency across prompt variations
- Multiple scoring strategies
- Category and difficulty analytics
- Phrasing sensitivity metric
- Dockerized execution
- Easily extensible dataset

---

# 📂 Project Structure

<<<<<<< HEAD
```text
=======
Each question has **exactly 3 phrasings**:
- **Phrasing 1** — Direct question
- **Phrasing 2** — Embedded / conversational phrasing
- **Phrasing 3** — Multiple-choice or reversed framing

## Metrics Computed

- **Overall accuracy** — fraction of all phrasings answered correctly
- **Category accuracy** — accuracy broken down by factual / math / logic
- **Difficulty accuracy** — accuracy broken down by easy / medium / hard
- **Phrasing sensitivity score** — standard deviation of per-question scores across 3 phrasings (0 = perfectly consistent, 0.47 = flipped one answer)
- **Per-question breakdown** — pass/fail for each of 3 phrasings

## Quick Start

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))
- Optional: an Anthropic API key if you want to use Claude instead

### 1. Clone and set up

```bash
git clone https://github.com/the-sadanand/llm-evaluation-harness
cd llm-eval-harness
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set LLM_PROVIDER=gemini and your GOOGLE_API_KEY
```

### 3. Run the full evaluation pipeline

```bash
python -m src.main
```

This runs all 3 steps automatically:
1. **Query** — sends 75 prompts (25 questions × 3 phrasings) to the target model
2. **Score** — applies exact match, numeric match, and LLM-as-judge scoring
3. **Analyze** — generates `results/report.json` and `results/summary_report.md`

### CLI Options

```bash
# Evaluate a specific model
python -m src.main --model claude-opus-4-6

# Use a custom dataset
python -m src.main --dataset path/to/dataset.json

# Skip re-querying (reuse existing raw_responses.json)
python -m src.main --skip-run

# Skip re-scoring (reuse existing scored_results.json)
python -m src.main --skip-score

# Adjust API call rate (default: 0.5s delay)
python -m src.main --delay 1.0

# Custom results directory
python -m src.main --results my_results/
```

## Docker

### Run with Docker Compose

```bash
cp .env.example .env    # fill in your API key
docker compose up
```

### Run tests in Docker

```bash
docker compose --profile test up test
```

### Build and run manually

```bash
docker build -t llm-eval-harness .
docker run --env-file .env -v $(pwd)/results:/app/results llm-eval-harness
```

## Output Files

After a successful run, the `results/` directory contains:

| File | Description |
|------|-------------|
| `raw_responses.json` | Raw model responses for every question+phrasing |
| `scored_results.json` | Responses annotated with scores and scoring method |
| `report.json` | Full structured report (aggregate + per-question breakdown) |
| `summary_report.md` | Human-readable Markdown summary report |

## Running Tests

```bash
pytest tests/ -v
```

All 32 tests cover:
- Exact match scoring (pass / fail / case-insensitive / punctuation)
- Numeric match scoring (integers / floats / comma-formatted / embedded in text)
- Score dispatch logic (correct strategy selected per answer type)
- Error response handling
- Dataset integrity (≥20 questions, 3 phrasings each, required fields, unique IDs)
- Analyzer statistics (accuracy, sensitivity, aggregate breakdowns)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `gemini` | Provider to use: `gemini` or `anthropic` |
| `GOOGLE_API_KEY` | ✅ Yes for Gemini | — | Your Google Gemini API key |
| `ANTHROPIC_API_KEY` | ✅ Yes for Anthropic | — | Your Anthropic API key |
| `TARGET_MODEL` | No | `gemini-2.0-flash` | Model to evaluate |
| `JUDGE_MODEL` | No | `gemini-2.0-flash` | Model used as LLM judge |
| `MAX_CALLS` | No | `0` | Limit the number of API calls; set to `10` or `20` to avoid free-tier exhaustion |
| `GEMINI_MAX_RETRIES` | No | `3` | Number of retry attempts after a rate-limit hit |
| `GEMINI_RETRY_DELAY` | No | `5` | Delay between Gemini retries in seconds |

## Project Structure

```
>>>>>>> 8268ecc190ea0e41619ff54334da5729f8306645
llm-eval-harness/
├── data/
├── src/
│   ├── main.py
│   ├── runner.py
│   ├── scorer.py
│   └── analyzer.py
├── tests/
├── results/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Quick Start

```bash
git clone https://github.com/the-sadanand/llm-evaluation-harness
cd llm-eval-harness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=YOUR_KEY
TARGET_MODEL=gemini-2.0-flash
```

Run

```bash
python -m src.main
```

<<<<<<< HEAD
---

# ⚙️ Pipeline

1. Load dataset
2. Generate prompt variants
3. Query target model
4. Collect responses
5. Score outputs
6. Compute statistics
7. Generate reports

---

# 📊 Scoring

| Strategy | Purpose |
|-----------|----------|
| Exact Match | Factual QA |
| Numeric Regex | Math |
| LLM Judge | Open-ended |
| Fallback Judge | Semantic verification |

---

# 📈 Metrics

- Overall Accuracy
- Category Accuracy
- Difficulty Accuracy
- Phrasing Sensitivity
- Per-question Consistency

---

# 🧪 Testing

```bash
pytest tests -v
```

Tests cover:

- Exact matching
- Numeric extraction
- Judge dispatch
- Dataset validation
- Analyzer statistics

---

# 🐳 Docker

```bash
docker compose up
```

---

# 🔧 Environment Variables

| Variable | Description |
|-----------|-------------|
| LLM_PROVIDER | gemini / anthropic |
| GOOGLE_API_KEY | Gemini key |
| ANTHROPIC_API_KEY | Claude key |
| TARGET_MODEL | Model under evaluation |
| JUDGE_MODEL | Judge model |

---

# 📄 Outputs

- raw_responses.json
- scored_results.json
- report.json
- summary_report.md

---

# 🛣 Roadmap

- Multi-turn evaluation
- Image benchmarks
- Code-generation benchmark
- Hallucination detection
- Cost analysis
- Latency benchmarking
- Web dashboard

---

# 🤝 Contributing

1. Fork
2. Branch
3. Commit
4. Push
5. Pull Request

---

# 👨‍💻 Author

**Sadanand Kumar**

Machine Learning Engineer • AI • LLMs • Computer Vision

GitHub: https://github.com/the-sadanand

LinkedIn: https://linkedin.com/in/sadanand-k7

---

## ⭐ If this project helped you, consider giving it a star!
=======
**Key finding:** Direct phrasings (phrasing_1) outperform multiple-choice/reversed phrasings (phrasing_3) by ~8%, suggesting the model is slightly more reliable when questions are posed directly rather than as constrained-choice problems.
>>>>>>> 8268ecc190ea0e41619ff54334da5729f8306645
