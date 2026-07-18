"""
LLM Evaluation Harness - Scoring Engine
Implements multiple scoring strategies:
  1. Exact / Fuzzy match (for numeric and factual answers)
  2. LLM-as-Judge (for open-ended or complex answers)
  3. Regex extraction (for multiple-choice and embedded answers)
"""

import os
import re
import json
import logging
from pathlib import Path

from src.providers import ProviderClient, resolve_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, judge_model: str = None, provider: str = None):
        self.provider = resolve_provider(provider)
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3.2:3b"
        self.provider_client = ProviderClient(provider=self.provider, model=self.judge_model)

    # ------------------------------------------------------------------
    # Strategy 1: Exact / Fuzzy Match
    # ------------------------------------------------------------------
    def exact_match(self, response: str, correct_answer: str) -> dict:
        """
        Normalize both strings and check for containment or equality.
        Returns score 1 (pass) or 0 (fail) plus explanation.
        """
        def normalize(text: str) -> str:
            text = text.lower().strip()
            # Remove common filler words and punctuation
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        norm_response = normalize(response)
        norm_answer = normalize(correct_answer)

        if norm_answer in norm_response or norm_response == norm_answer:
            return {"score": 1, "method": "exact_match", "explanation": "Answer found in response."}

        return {"score": 0, "method": "exact_match", "explanation": f"Expected '{correct_answer}', got '{response[:80]}'."}

    # ------------------------------------------------------------------
    # Strategy 2: Numeric / Regex Match
    # ------------------------------------------------------------------
    def numeric_match(self, response: str, correct_answer: str) -> dict:
        """
        Extract all numbers from the response and check if the correct numeric
        answer appears among them. Falls back to exact match for non-numeric answers.
        """
        try:
            expected = float(str(correct_answer).replace(",", "").strip())
        except ValueError:
            return self.exact_match(response, correct_answer)

        # Extract all numbers from the model response
        numbers = re.findall(r"-?\d+(?:\.\d+)?", response.replace(",", ""))
        found_numbers = [float(n) for n in numbers]

        if expected in found_numbers:
            return {
                "score": 1,
                "method": "numeric_match",
                "explanation": f"Correct number {expected} found in response."
            }

        # Close-enough check (within 0.01% for floating-point quirks)
        for n in found_numbers:
            if abs(n - expected) / max(abs(expected), 1e-9) < 0.001:
                return {
                    "score": 1,
                    "method": "numeric_match",
                    "explanation": f"Near-exact numeric match: {n} ≈ {expected}."
                }

        return {
            "score": 0,
            "method": "numeric_match",
            "explanation": f"Expected {expected}, found numbers: {found_numbers[:5]} in response."
        }

    # ------------------------------------------------------------------
    # Strategy 3: LLM-as-Judge
    # ------------------------------------------------------------------
    def llm_judge(self, question: str, response: str, correct_answer: str) -> dict:
        """
        Use a powerful LLM to rate the response on a scale of 1-5 and return a
        binary pass/fail (>= 3 = pass) plus the raw score and justification.
        """
        prompt = f"""You are an objective evaluator. Rate how well the model's response answers the question, given the correct answer.

Question: {question}
Correct Answer: {correct_answer}
Model Response: {response}

Scoring rubric:
5 - Perfectly correct and complete
4 - Correct with minor imprecision
3 - Partially correct (key information present)
2 - Mostly wrong but shows understanding
1 - Completely wrong or irrelevant

Respond ONLY in this exact JSON format (no markdown, no extra text):
{{"score": <integer 1-5>, "justification": "<one sentence>"}}"""

        try:
            if not self.provider_client.is_available():
                raise RuntimeError(self.provider_client.get_error() or "Provider is not configured")
            raw = self.provider_client.generate_text(prompt, max_tokens=256)
            result = json.loads(raw)
            rubric_score = int(result["score"])
            binary = 1 if rubric_score >= 3 else 0
            return {
                "score": binary,
                "rubric_score": rubric_score,
                "method": "llm_judge",
                "explanation": result.get("justification", "")
            }
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}. Falling back to exact match.")
            return self.exact_match(response, correct_answer)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def score(self, record: dict) -> dict:
        """
        Choose the right scoring strategy based on answer_type and return enriched record.
        """
        answer_type = record.get("answer_type", "exact")
        response = record["model_response"]
        correct = record["correct_answer"]

        if response.startswith("ERROR:"):
            result = {"score": 0, "method": "error", "explanation": response}
        elif answer_type == "numeric":
            result = self.numeric_match(response, correct)
        elif answer_type == "open_ended":
            result = self.llm_judge(record["phrasing_text"], response, correct)
        else:
            # Default: try exact match first; if it fails, try LLM judge
            exact = self.exact_match(response, correct)
            if exact["score"] == 1:
                result = exact
            else:
                # Use LLM judge as a fallback for factual/logic questions
                result = self.llm_judge(record["phrasing_text"], response, correct)

        return {**record, **result}

    def score_all(self, raw_results: list) -> list:
        """Score every record in the raw results list."""
        scored = []
        for i, record in enumerate(raw_results):
            logger.info(f"Scoring [{i+1}/{len(raw_results)}] {record['question_id']} | {record['phrasing_label']}")
            scored.append(self.score(record))
        return scored


if __name__ == "__main__":
    results_dir = Path("results")
    raw_path = results_dir / "raw_responses.json"

    with open(raw_path) as f:
        raw = json.load(f)

    scorer = Scorer()
    scored = scorer.score_all(raw)

    out_path = results_dir / "scored_results.json"
    with open(out_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"Scored results saved to {out_path}")
