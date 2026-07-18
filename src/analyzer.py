"""
LLM Evaluation Harness - Analyzer & Report Generator
Computes per-question statistics, phrasing sensitivity, category breakdowns,
and generates both a JSON results file and a Markdown summary report.
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def std_dev(values: list) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return round(math.sqrt(variance), 4)


class Analyzer:
    def __init__(self, scored_results: list, dataset: list):
        self.scored = scored_results
        self.dataset = {q["id"]: q for q in dataset}

    def _group_by_question(self) -> dict:
        """Group scored records by question ID."""
        groups = defaultdict(list)
        for record in self.scored:
            groups[record["question_id"]].append(record)
        return dict(groups)

    def compute_per_question_stats(self) -> list:
        """
        For each question, compute:
          - scores per phrasing
          - accuracy (fraction of phrasings answered correctly)
          - phrasing sensitivity score (std dev of scores)
        """
        groups = self._group_by_question()
        stats = []

        for qid, records in groups.items():
            # Sort by phrasing index to keep order stable
            records_sorted = sorted(records, key=lambda r: r["phrasing_index"])
            scores = [r["score"] for r in records_sorted]
            accuracy = round(sum(scores) / len(scores), 4) if scores else 0.0
            sensitivity = std_dev(scores)

            phrasing_detail = []
            for r in records_sorted:
                phrasing_detail.append({
                    "phrasing_index": r["phrasing_index"],
                    "phrasing_text": r["phrasing_text"],
                    "model_response": r["model_response"],
                    "score": r["score"],
                    "method": r.get("method", "unknown"),
                    "explanation": r.get("explanation", ""),
                    "rubric_score": r.get("rubric_score", None)
                })

            stats.append({
                "question_id": qid,
                "category": records_sorted[0]["category"],
                "difficulty": records_sorted[0]["difficulty"],
                "answer_type": records_sorted[0]["answer_type"],
                "correct_answer": records_sorted[0]["correct_answer"],
                "scores_per_phrasing": scores,
                "accuracy": accuracy,
                "phrasing_sensitivity": sensitivity,
                "passed_all": all(s == 1 for s in scores),
                "failed_all": all(s == 0 for s in scores),
                "phrasings": phrasing_detail
            })

        # Sort by question ID
        stats.sort(key=lambda x: x["question_id"])
        return stats

    def compute_aggregate_stats(self, per_question: list) -> dict:
        """Aggregate metrics across all questions."""
        all_scores = [s for q in per_question for s in q["scores_per_phrasing"]]
        overall_accuracy = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

        # By category
        by_category = defaultdict(list)
        for q in per_question:
            by_category[q["category"]].extend(q["scores_per_phrasing"])

        category_accuracy = {
            cat: round(sum(scores) / len(scores), 4)
            for cat, scores in by_category.items()
        }

        # By difficulty
        by_difficulty = defaultdict(list)
        for q in per_question:
            by_difficulty[q["difficulty"]].extend(q["scores_per_phrasing"])

        difficulty_accuracy = {
            diff: round(sum(scores) / len(scores), 4)
            for diff, scores in by_difficulty.items()
        }

        # Most flaky (high sensitivity, not perfect)
        flaky = sorted(
            [q for q in per_question if not q["passed_all"]],
            key=lambda x: x["phrasing_sensitivity"],
            reverse=True
        )

        # Most frequently failed (zero accuracy)
        most_failed = [q for q in per_question if q["accuracy"] == 0.0]

        # Phrasing direction analysis
        phrasing_scores = defaultdict(list)
        for q in per_question:
            for i, score in enumerate(q["scores_per_phrasing"]):
                phrasing_scores[f"phrasing_{i+1}"].append(score)

        phrasing_accuracy = {
            label: round(sum(scores) / len(scores), 4)
            for label, scores in phrasing_scores.items()
        }

        return {
            "overall_accuracy": overall_accuracy,
            "total_questions": len(per_question),
            "total_phrasings_tested": len(all_scores),
            "questions_passed_all": sum(1 for q in per_question if q["passed_all"]),
            "questions_failed_all": sum(1 for q in per_question if q["failed_all"]),
            "category_accuracy": category_accuracy,
            "difficulty_accuracy": difficulty_accuracy,
            "phrasing_accuracy": phrasing_accuracy,
            "avg_phrasing_sensitivity": round(
                sum(q["phrasing_sensitivity"] for q in per_question) / len(per_question), 4
            ) if per_question else 0.0,
            "most_flaky_questions": [q["question_id"] for q in flaky[:5]],
            "most_failed_questions": [q["question_id"] for q in most_failed],
        }

    def generate_report(self, per_question: list, aggregate: dict, output_path: Path):
        """Write a Markdown summary report."""
        lines = []
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        lines.append("# LLM Evaluation Harness — Summary Report")
        lines.append(f"\n**Generated:** {ts}\n")
        lines.append("---\n")

        # --- Executive Summary ---
        lines.append("## Executive Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Overall Accuracy | **{aggregate['overall_accuracy']*100:.1f}%** |")
        lines.append(f"| Total Questions | {aggregate['total_questions']} |")
        lines.append(f"| Total Phrasings Tested | {aggregate['total_phrasings_tested']} |")
        lines.append(f"| Questions Passed All 3 Phrasings | {aggregate['questions_passed_all']} |")
        lines.append(f"| Questions Failed All 3 Phrasings | {aggregate['questions_failed_all']} |")
        lines.append(f"| Avg Phrasing Sensitivity (Std Dev) | {aggregate['avg_phrasing_sensitivity']:.4f} |")
        lines.append("")

        # --- Phrasing Sensitivity Analysis ---
        lines.append("## Phrasing Sensitivity Analysis\n")
        pa = aggregate["phrasing_accuracy"]
        lines.append("Accuracy by phrasing variant (phrasing_1 = direct, phrasing_2 = embedded, phrasing_3 = multiple-choice/reversed):\n")
        lines.append("| Phrasing | Accuracy |")
        lines.append("|----------|---------|")
        for label in sorted(pa.keys()):
            lines.append(f"| {label} | {pa[label]*100:.1f}% |")
        lines.append("")

        # Interpretation
        vals = list(pa.values())
        sensitivity_spread = max(vals) - min(vals) if vals else 0
        lines.append(f"> **Phrasing sensitivity spread:** {sensitivity_spread*100:.1f}% difference between best and worst phrasing variant.")
        if sensitivity_spread > 0.15:
            lines.append("> ⚠️  High phrasing sensitivity detected — the model's answers change significantly based on how questions are phrased.")
        else:
            lines.append("> ✅  Low phrasing sensitivity — the model is relatively robust to rephrasing.")
        lines.append("")

        # --- Category Breakdown ---
        lines.append("## Accuracy by Category\n")
        lines.append("| Category | Accuracy |")
        lines.append("|----------|---------|")
        for cat, acc in sorted(aggregate["category_accuracy"].items()):
            lines.append(f"| {cat} | {acc*100:.1f}% |")
        lines.append("")

        # --- Difficulty Breakdown ---
        lines.append("## Accuracy by Difficulty\n")
        lines.append("| Difficulty | Accuracy |")
        lines.append("|------------|---------|")
        for diff, acc in sorted(aggregate["difficulty_accuracy"].items()):
            lines.append(f"| {diff} | {acc*100:.1f}% |")
        lines.append("")

        # --- Most Failed Questions ---
        lines.append("## Most Frequently Failed Questions\n")
        if aggregate["most_failed_questions"]:
            for qid in aggregate["most_failed_questions"]:
                q = next((x for x in per_question if x["question_id"] == qid), None)
                if q:
                    lines.append(f"- **{qid}** ({q['category']}, {q['difficulty']}): `{q['correct_answer']}` — failed all phrasings")
        else:
            lines.append("_No questions failed all three phrasings._")
        lines.append("")

        # --- Most Flaky Questions ---
        lines.append("## Most Phrasing-Sensitive (Flaky) Questions\n")
        if aggregate["most_flaky_questions"]:
            for qid in aggregate["most_flaky_questions"]:
                q = next((x for x in per_question if x["question_id"] == qid), None)
                if q:
                    lines.append(f"- **{qid}** sensitivity={q['phrasing_sensitivity']:.4f} | scores={q['scores_per_phrasing']}")
        else:
            lines.append("_No flaky questions detected._")
        lines.append("")

        # --- Per-Question Breakdown ---
        lines.append("## Per-Question Breakdown\n")
        lines.append("| Q ID | Category | Difficulty | P1 | P2 | P3 | Accuracy | Sensitivity |")
        lines.append("|------|----------|------------|----|----|----|---------:|------------:|")
        for q in per_question:
            sc = q["scores_per_phrasing"]
            p1 = "✅" if len(sc) > 0 and sc[0] == 1 else "❌"
            p2 = "✅" if len(sc) > 1 and sc[1] == 1 else "❌"
            p3 = "✅" if len(sc) > 2 and sc[2] == 1 else "❌"
            lines.append(
                f"| {q['question_id']} | {q['category']} | {q['difficulty']} | "
                f"{p1} | {p2} | {p3} | {q['accuracy']*100:.0f}% | {q['phrasing_sensitivity']:.4f} |"
            )
        lines.append("")

        # --- Key Insights ---
        lines.append("## Key Insights\n")
        best_cat = max(aggregate["category_accuracy"], key=aggregate["category_accuracy"].get)
        worst_cat = min(aggregate["category_accuracy"], key=aggregate["category_accuracy"].get)
        lines.append(f"1. **Best performing category:** `{best_cat}` at {aggregate['category_accuracy'][best_cat]*100:.1f}% accuracy.")
        lines.append(f"2. **Worst performing category:** `{worst_cat}` at {aggregate['category_accuracy'][worst_cat]*100:.1f}% accuracy.")

        best_phrasing = max(pa, key=pa.get)
        worst_phrasing = min(pa, key=pa.get)
        lines.append(f"3. **Phrasing direction effect:** `{best_phrasing}` outperforms `{worst_phrasing}` by {(pa[best_phrasing]-pa[worst_phrasing])*100:.1f}%.")

        if aggregate["avg_phrasing_sensitivity"] > 0.3:
            lines.append("4. **High overall phrasing sensitivity** — consider prompt standardization in production to reduce variance.")
        else:
            lines.append("4. **Low overall phrasing sensitivity** — the model handles rephrasing well across most question types.")
        lines.append("")

        lines.append("---")
        lines.append("_Report generated by LLM Evaluation Harness v1.0_")

        output_path.write_text("\n".join(lines))
        logger.info(f"Summary report saved to {output_path}")


def run_analysis(scored_path: Path, dataset_path: Path, results_dir: Path):
    with open(scored_path) as f:
        scored = json.load(f)
    with open(dataset_path) as f:
        dataset = json.load(f)

    analyzer = Analyzer(scored, dataset)
    per_question = analyzer.compute_per_question_stats()
    aggregate = analyzer.compute_aggregate_stats(per_question)

    # Save detailed per-question report
    report_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "aggregate": aggregate,
        "per_question": per_question
    }
    report_json_path = results_dir / "report.json"
    with open(report_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Detailed JSON report saved to {report_json_path}")

    # Save Markdown summary
    summary_path = results_dir / "summary_report.md"
    analyzer.generate_report(per_question, aggregate, summary_path)

    return report_data


if __name__ == "__main__":
    results_dir = Path("results")
    run_analysis(
        scored_path=results_dir / "scored_results.json",
        dataset_path=Path("data/dataset.json"),
        results_dir=results_dir
    )
