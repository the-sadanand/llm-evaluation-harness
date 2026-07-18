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

```text
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
