import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import EvalRunner


def test_partial_results_are_saved_when_max_calls_is_reached(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.json"
    results_dir = tmp_path / "results"
    dataset_path.write_text(
        json.dumps([
            {
                "id": "Q001",
                "category": "factual",
                "difficulty": "easy",
                "correct_answer": "Paris",
                "answer_type": "exact",
                "phrasings": ["What is the capital?", "Which city is the capital?"],
            }
        ])
    )

    monkeypatch.setenv("MAX_CALLS", "2")
    runner = EvalRunner(str(dataset_path), str(results_dir), model="dummy", provider="ollama")
    monkeypatch.setattr(runner, "_query_model", lambda prompt: "Paris")

    raw_results = runner.run(delay=0)

    assert len(raw_results) == 2
    saved_results = json.loads((results_dir / "raw_responses.json").read_text())
    assert saved_results == raw_results
