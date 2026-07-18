"""
LLM Evaluation Harness - Orchestrator
Runs the full pipeline: dataset → query → score → analyze → report
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency fallback
    def load_dotenv() -> bool:
        return False

load_dotenv(override=True)

# Add project root to path so imports work regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import EvalRunner
from src.scorer import Scorer
from src.analyzer import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_env(provider: str | None = None):
    """Validate the local Ollama configuration."""
    selected_provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if selected_provider == "ollama":
        if not os.getenv("OLLAMA_BASE_URL"):
            logger.warning("OLLAMA_BASE_URL is not set; using the default http://127.0.0.1:11434")
        return

    logger.error("This project only supports the Ollama provider.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Harness")
    parser.add_argument("--dataset", default="data/dataset.json", help="Path to dataset JSON")
    parser.add_argument("--results", default="results", help="Output directory for results")
    parser.add_argument("--model", default=None, help="Target model to evaluate (overrides TARGET_MODEL env var)")
    parser.add_argument("--provider", default="ollama", choices=["ollama"], help="LLM provider to use")
    parser.add_argument("--judge-model", default=None, help="Judge model to use for LLM-as-judge scoring")
    parser.add_argument("--max-calls", type=int, default=None, help="Limit the number of prompts sent to the provider")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between API calls")
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip querying the model (use existing raw_responses.json)"
    )
    parser.add_argument(
        "--skip-score", action="store_true",
        help="Skip scoring (use existing scored_results.json)"
    )
    args = parser.parse_args()

    if args.max_calls is not None:
        os.environ["MAX_CALLS"] = str(args.max_calls)

    check_env(provider=args.provider)

    dataset_path = Path(args.dataset)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_path = results_dir / "raw_responses.json"
    scored_path = results_dir / "scored_results.json"

    # ── Step 1: Run the model ──────────────────────────────────────────
    if args.skip_run and raw_path.exists():
        logger.info(f"Skipping model run, loading existing {raw_path}")
        with open(raw_path) as f:
            raw_results = json.load(f)
    else:
        logger.info("=== Step 1: Querying target LLM ===")
        runner = EvalRunner(
            dataset_path=str(dataset_path),
            results_dir=str(results_dir),
            model=args.model,
            provider=args.provider
        )
        raw_results = runner.run(delay=args.delay)

    # ── Step 2: Score responses ────────────────────────────────────────
    if args.skip_score and scored_path.exists():
        logger.info(f"Skipping scoring, loading existing {scored_path}")
        with open(scored_path) as f:
            scored_results = json.load(f)
    else:
        logger.info("=== Step 2: Scoring responses ===")
        scorer = Scorer(judge_model=args.judge_model, provider=args.provider)
        scored_results = scorer.score_all(raw_results)
        with open(scored_path, "w") as f:
            json.dump(scored_results, f, indent=2)
        logger.info(f"Scored results saved to {scored_path}")

    # ── Step 3: Analyze & report ───────────────────────────────────────
    logger.info("=== Step 3: Analyzing results and generating reports ===")
    report = run_analysis(
        scored_path=scored_path,
        dataset_path=dataset_path,
        results_dir=results_dir
    )

    # Print summary to console
    agg = report["aggregate"]
    print("\n" + "="*60)
    print("  LLM EVALUATION HARNESS — RESULTS SUMMARY")
    print("="*60)
    print(f"  Overall Accuracy:      {agg['overall_accuracy']*100:.1f}%")
    print(f"  Questions Evaluated:   {agg['total_questions']}")
    print(f"  Phrasings Tested:      {agg['total_phrasings_tested']}")
    print(f"  Passed All Phrasings:  {agg['questions_passed_all']}")
    print(f"  Failed All Phrasings:  {agg['questions_failed_all']}")
    print(f"  Avg Sensitivity Score: {agg['avg_phrasing_sensitivity']:.4f}")
    print("\n  Category Accuracy:")
    for cat, acc in sorted(agg["category_accuracy"].items()):
        print(f"    {cat:<15} {acc*100:.1f}%")
    print("\n  Outputs written to:")
    print(f"    {results_dir}/raw_responses.json")
    print(f"    {results_dir}/scored_results.json")
    print(f"    {results_dir}/report.json")
    print(f"    {results_dir}/summary_report.md")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
