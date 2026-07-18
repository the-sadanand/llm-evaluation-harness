"""
LLM Evaluation Harness - Main Runner
Queries the target LLM for each question/phrasing and stores raw responses.
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

from src.providers import ProviderClient, resolve_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, dataset_path: str, results_dir: str, model: str = None, provider: str = None):
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.provider = resolve_provider(provider)
        self.model = model or os.getenv("TARGET_MODEL") or (
            "gemini-2.0-flash" if self.provider == "gemini" else "claude-haiku-4-5-20251001"
        )
        self.provider_client = ProviderClient(provider=self.provider, model=self.model)
        self.dataset = self._load_dataset()
        self.max_calls = int(os.getenv("MAX_CALLS", "0"))

    def _load_dataset(self):
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} questions from {self.dataset_path}")
        return data

    def _query_model(self, prompt: str) -> str:
        """Send a prompt to the target LLM and return the raw response text."""
        try:
            if not self.provider_client.is_available():
                error = self.provider_client.get_error() or "Provider is not configured"
                logger.error(f"Provider unavailable: {error}")
                return f"ERROR: {error}"
            return self.provider_client.generate_text(prompt, max_tokens=512)
        except Exception as e:
            logger.error(f"API error: {e}")
            return f"ERROR: {str(e)}"

    def run(self, delay: float = 0.5) -> list:
        """
        Iterate through all questions and phrasings, query the model, and collect results.
        Returns a list of raw result records.
        """
        raw_results = []
        total = sum(len(q["phrasings"]) for q in self.dataset)
        count = 0

        for question in self.dataset:
            qid = question["id"]
            for idx, phrasing in enumerate(question["phrasings"]):
                if self.max_calls and count >= self.max_calls:
                    logger.warning("Reached MAX_CALLS=%s; stopping early.", self.max_calls)
                    return raw_results

                count += 1
                phrasing_label = f"phrasing_{idx + 1}"
                logger.info(f"[{count}/{total}] {qid} | {phrasing_label}")

                response = self._query_model(phrasing)

                raw_results.append({
                    "question_id": qid,
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "answer_type": question["answer_type"],
                    "correct_answer": question["correct_answer"],
                    "phrasing_index": idx + 1,
                    "phrasing_label": phrasing_label,
                    "phrasing_text": phrasing,
                    "model_response": response,
                    "model": self.model,
                    "provider": self.provider,
                    "timestamp": datetime.utcnow().isoformat()
                })

                if delay:
                    time.sleep(delay)

        # Save raw responses
        raw_path = self.results_dir / "raw_responses.json"
        with open(raw_path, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Raw responses saved to {raw_path}")

        return raw_results


if __name__ == "__main__":
    runner = EvalRunner(
        dataset_path="data/dataset.json",
        results_dir="results"
    )
    runner.run()
