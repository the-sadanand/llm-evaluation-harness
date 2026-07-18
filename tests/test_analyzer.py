"""
Unit tests for the Analyzer module.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import Analyzer, std_dev


class TestStdDev:
    def test_all_same(self):
        assert std_dev([1, 1, 1]) == 0.0

    def test_perfect_variance(self):
        result = std_dev([1, 0, 1])
        assert abs(result - 0.4714) < 0.001

    def test_single_value(self):
        assert std_dev([1]) == 0.0

    def test_zeros(self):
        assert std_dev([0, 0, 0]) == 0.0

    def test_max_variance(self):
        result = std_dev([1, 0])
        assert abs(result - 0.5) < 0.001


class TestAnalyzer:
    def _make_dataset(self):
        return [
            {
                "id": "Q001", "category": "factual", "difficulty": "easy",
                "correct_answer": "Paris", "answer_type": "exact",
                "phrasings": ["P1", "P2", "P3"]
            },
            {
                "id": "Q002", "category": "math", "difficulty": "hard",
                "correct_answer": "150", "answer_type": "numeric",
                "phrasings": ["P1", "P2", "P3"]
            },
        ]

    def _make_scored(self):
        return [
            # Q001 — passes all phrasings
            {"question_id": "Q001", "category": "factual", "difficulty": "easy",
             "answer_type": "exact", "correct_answer": "Paris",
             "phrasing_index": 1, "phrasing_label": "phrasing_1",
             "phrasing_text": "What is the capital?",
             "model_response": "Paris", "score": 1, "method": "exact_match", "explanation": ""},
            {"question_id": "Q001", "category": "factual", "difficulty": "easy",
             "answer_type": "exact", "correct_answer": "Paris",
             "phrasing_index": 2, "phrasing_label": "phrasing_2",
             "phrasing_text": "Which city is the capital?",
             "model_response": "The capital is Paris.", "score": 1, "method": "exact_match", "explanation": ""},
            {"question_id": "Q001", "category": "factual", "difficulty": "easy",
             "answer_type": "exact", "correct_answer": "Paris",
             "phrasing_index": 3, "phrasing_label": "phrasing_3",
             "phrasing_text": "Paris or London?",
             "model_response": "Paris is the answer.", "score": 1, "method": "exact_match", "explanation": ""},

            # Q002 — fails all phrasings
            {"question_id": "Q002", "category": "math", "difficulty": "hard",
             "answer_type": "numeric", "correct_answer": "150",
             "phrasing_index": 1, "phrasing_label": "phrasing_1",
             "phrasing_text": "60 * 2.5 = ?",
             "model_response": "I think it is 200.", "score": 0, "method": "numeric_match", "explanation": ""},
            {"question_id": "Q002", "category": "math", "difficulty": "hard",
             "answer_type": "numeric", "correct_answer": "150",
             "phrasing_index": 2, "phrasing_label": "phrasing_2",
             "phrasing_text": "Distance question",
             "model_response": "The answer is 120.", "score": 0, "method": "numeric_match", "explanation": ""},
            {"question_id": "Q002", "category": "math", "difficulty": "hard",
             "answer_type": "numeric", "correct_answer": "150",
             "phrasing_index": 3, "phrasing_label": "phrasing_3",
             "phrasing_text": "Calculate distance",
             "model_response": "90 miles.", "score": 0, "method": "numeric_match", "explanation": ""},
        ]

    def test_per_question_accuracy(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        stats = analyzer.compute_per_question_stats()
        q1 = next(q for q in stats if q["question_id"] == "Q001")
        q2 = next(q for q in stats if q["question_id"] == "Q002")
        assert q1["accuracy"] == 1.0
        assert q2["accuracy"] == 0.0

    def test_passed_all(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        stats = analyzer.compute_per_question_stats()
        q1 = next(q for q in stats if q["question_id"] == "Q001")
        assert q1["passed_all"] is True
        assert q1["failed_all"] is False

    def test_failed_all(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        stats = analyzer.compute_per_question_stats()
        q2 = next(q for q in stats if q["question_id"] == "Q002")
        assert q2["failed_all"] is True
        assert q2["passed_all"] is False

    def test_phrasing_sensitivity_zero_for_perfect(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        stats = analyzer.compute_per_question_stats()
        q1 = next(q for q in stats if q["question_id"] == "Q001")
        assert q1["phrasing_sensitivity"] == 0.0

    def test_aggregate_overall_accuracy(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        per_question = analyzer.compute_per_question_stats()
        agg = analyzer.compute_aggregate_stats(per_question)
        # Q001: 3/3, Q002: 0/3 → 3/6 = 0.5
        assert agg["overall_accuracy"] == 0.5

    def test_aggregate_category_accuracy(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        per_question = analyzer.compute_per_question_stats()
        agg = analyzer.compute_aggregate_stats(per_question)
        assert agg["category_accuracy"]["factual"] == 1.0
        assert agg["category_accuracy"]["math"] == 0.0

    def test_most_failed_questions(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        per_question = analyzer.compute_per_question_stats()
        agg = analyzer.compute_aggregate_stats(per_question)
        assert "Q002" in agg["most_failed_questions"]

    def test_phrasing_count_in_detail(self):
        analyzer = Analyzer(self._make_scored(), self._make_dataset())
        stats = analyzer.compute_per_question_stats()
        for q in stats:
            assert len(q["phrasings"]) == 3
