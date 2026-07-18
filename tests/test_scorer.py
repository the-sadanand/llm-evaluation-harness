"""
Unit tests for the Scorer module.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
from src.scorer import Scorer


class TestExactMatch:
    def setup_method(self):
        self.scorer = Scorer()

    def test_exact_match_pass(self):
        result = self.scorer.exact_match("The answer is Paris.", "Paris")
        assert result["score"] == 1

    def test_exact_match_fail(self):
        result = self.scorer.exact_match("The answer is London.", "Paris")
        assert result["score"] == 0

    def test_exact_match_case_insensitive(self):
        result = self.scorer.exact_match("PARIS is the capital.", "paris")
        assert result["score"] == 1

    def test_exact_match_with_punctuation(self):
        result = self.scorer.exact_match("The answer is Paris!", "Paris")
        assert result["score"] == 1

    def test_exact_match_embedded_in_response(self):
        result = self.scorer.exact_match(
            "Australia's capital is Canberra, not Sydney.", "Canberra"
        )
        assert result["score"] == 1


class TestNumericMatch:
    def setup_method(self):
        self.scorer = Scorer()

    def test_numeric_match_integer(self):
        result = self.scorer.numeric_match("The answer is 150 miles.", "150")
        assert result["score"] == 1

    def test_numeric_match_fail(self):
        result = self.scorer.numeric_match("I think it's 200 miles.", "150")
        assert result["score"] == 0

    def test_numeric_match_with_units(self):
        result = self.scorer.numeric_match("The distance is 150km.", "150")
        assert result["score"] == 1

    def test_numeric_match_in_sentence(self):
        result = self.scorer.numeric_match(
            "Multiplying 60 by 2.5 gives us 150 miles total.", "150"
        )
        assert result["score"] == 1

    def test_numeric_match_commas_in_number(self):
        result = self.scorer.numeric_match("The result is 1,000.", "1000")
        assert result["score"] == 1

    def test_numeric_match_float(self):
        result = self.scorer.numeric_match("The answer is 3.14.", "3.14")
        assert result["score"] == 1


class TestScoreDispatch:
    def setup_method(self):
        self.scorer = Scorer()

    def test_dispatch_numeric(self):
        record = {
            "question_id": "Q001",
            "phrasing_text": "What is 60 * 2.5?",
            "model_response": "The answer is 150 miles.",
            "correct_answer": "150",
            "answer_type": "numeric"
        }
        result = self.scorer.score(record)
        assert result["score"] == 1
        assert result["method"] == "numeric_match"

    def test_dispatch_error_response(self):
        record = {
            "question_id": "Q001",
            "phrasing_text": "What is the capital?",
            "model_response": "ERROR: API timeout",
            "correct_answer": "Paris",
            "answer_type": "exact"
        }
        result = self.scorer.score(record)
        assert result["score"] == 0
        assert result["method"] == "error"

    def test_exact_dispatch_with_correct_answer(self):
        record = {
            "question_id": "Q005",
            "phrasing_text": "What is the capital of France?",
            "model_response": "The capital of France is Paris.",
            "correct_answer": "Paris",
            "answer_type": "exact"
        }
        result = self.scorer.score(record)
        assert result["score"] == 1


class TestDatasetIntegrity:
    """Validate that the dataset meets project requirements."""

    def test_dataset_has_minimum_questions(self):
        import json
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        assert len(dataset) >= 20, f"Dataset has only {len(dataset)} questions, need >= 20"

    def test_each_question_has_three_phrasings(self):
        import json
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        for q in dataset:
            assert len(q["phrasings"]) == 3, (
                f"Question {q['id']} has {len(q['phrasings'])} phrasings, expected 3"
            )

    def test_each_question_has_correct_answer(self):
        import json
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        for q in dataset:
            assert "correct_answer" in q and q["correct_answer"], (
                f"Question {q['id']} is missing correct_answer"
            )

    def test_question_ids_are_unique(self):
        import json
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        ids = [q["id"] for q in dataset]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_required_fields_present(self):
        import json
        with open("data/dataset.json") as f:
            dataset = json.load(f)
        required_fields = {"id", "category", "difficulty", "correct_answer", "answer_type", "phrasings"}
        for q in dataset:
            missing = required_fields - set(q.keys())
            assert not missing, f"Question {q['id']} missing fields: {missing}"
