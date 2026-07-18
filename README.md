# LLM Evaluation Harness

A production-grade evaluation framework to benchmark LLM reliability, consistency, and phrasing sensitivity across diverse question types.

## Overview

This harness measures whether a language model answers **consistently** across semantically equivalent but lexically different phrasings of the same question — a critical metric for production AI systems.

### Evaluation Architecture

```
Question Dataset          Phrasing          LLM Under       Response      Scoring
25 items with    →   Generator (3    →     Test      →   Collector →   Engine
ground truth          variants/q)        (via API)                      ↓
                                                                   ┌─────────────┐
                                                                   │ Exact Match │ → Score 1/0
                                                                   │ Numeric     │ → Score 1/0
                                                                   │ LLM Judge   │ → Rubric 1-5
                                                                   └─────────────┘
                                                                         ↓
                                                              Results DB (JSON)
                                                                         ↓
                                                              Analysis & Report
                                                            (report.json, summary_report.md)
```

### Scoring Strategies

| Answer Type | Method | Description |
|-------------|--------|-------------|
| Exact / Factual | **Exact Match** | Case-insensitive containment check with punctuation normalization |
| Numeric / Math | **Numeric Regex Match** | Extracts all numbers from response and checks for correct value |
| Open-ended | **LLM-as-Judge** | Uses a judge model to rate response 1–5 with justification; ≥3 = pass |
| Factual fallback | **LLM-as-Judge** | Falls back to LLM judge when exact match fails, catching paraphrased correct answers |

## Dataset

25 unique questions across 4 categories and 3 difficulty levels:

| Category | Count | Difficulty |
|----------|-------|-----------|
| Factual  | 13    | Easy / Medium / Hard |
| Math     | 9     | Easy / Medium / Hard |
| Logic    | 3     | Medium / Hard |

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
git clone <your-repo-url>
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
llm-eval-harness/
├── data/
│   └── dataset.json          # 25 questions with 3 phrasings each
├── results/                  # Generated at runtime
│   ├── raw_responses.json
│   ├── scored_results.json
│   ├── report.json
│   └── summary_report.md
├── src/
│   ├── __init__.py
│   ├── main.py               # Orchestrator — runs the full pipeline
│   ├── runner.py             # Queries the target LLM
│   ├── scorer.py             # Scoring strategies (exact, numeric, LLM-judge)
│   └── analyzer.py           # Statistics, aggregation, report generation
├── tests/
│   ├── __init__.py
│   ├── test_scorer.py        # Unit tests for scoring
│   └── test_analyzer.py      # Unit tests for analysis
├── .env.example              # Required environment variables
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Extending the Dataset

Add more questions to `data/dataset.json` following this schema:

```json
{
  "id": "Q026",
  "category": "factual",
  "difficulty": "medium",
  "correct_answer": "Marie Curie",
  "answer_type": "exact",
  "phrasings": [
    "Who was the first woman to win a Nobel Prize?",
    "Which scientist became the first female Nobel laureate?",
    "Marie Curie, Florence Nightingale, or Rosalind Franklin — who was first to win a Nobel Prize?"
  ]
}
```

Supported `answer_type` values: `exact`, `numeric`, `open_ended`.

## Sample Results

From a sample run against `claude-haiku-4-5-20251001`:

| Metric | Value |
|--------|-------|
| Overall Accuracy | 92.0% |
| Math Accuracy | 96.7% |
| Factual Accuracy | 91.7% |
| Logic Accuracy | 77.8% |
| Phrasing Sensitivity (avg) | 0.1131 |

**Key finding:** Direct phrasings (phrasing_1) outperform multiple-choice/reversed phrasings (phrasing_3) by ~8%, suggesting the model is slightly more reliable when questions are posed directly rather than as constrained-choice problems.

## Security Notes

- **Never commit your `.env` file** — it is in `.gitignore`
- LLM judge calls use a separate model to avoid self-evaluation bias
- API keys are read from environment variables only, never hardcoded
