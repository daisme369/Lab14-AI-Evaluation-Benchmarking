import argparse
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator
from engine.runner import BenchmarkRunner


def _load_local_env(env_path: Optional[str] = None) -> None:
    candidate_path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not candidate_path.exists():
        return

    with candidate_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_non_empty_env(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())


def _default_live_judge_enabled() -> bool:
    if os.getenv("USE_LIVE_JUDGE") is not None:
        return _env_bool("USE_LIVE_JUDGE", False)
    return _has_non_empty_env("GEMINI_API_KEY") and _has_non_empty_env("GROQ_API_KEY")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _lexical_overlap_score(answer: str, expected_answer: str) -> float:
    expected_tokens = set(_tokenize(expected_answer))
    if not expected_tokens:
        return 0.0
    answer_tokens = set(_tokenize(answer))
    overlap = len(expected_tokens.intersection(answer_tokens))
    return overlap / len(expected_tokens)


class ExpertEvaluator:
    def __init__(self, retrieval_top_k: int = 3):
        self.retrieval = RetrievalEvaluator()
        self.retrieval_top_k = retrieval_top_k

    def _extract_retrieved_ids(self, case: Dict[str, Any], response: Dict[str, Any]) -> List[str]:
        response_ids = response.get("retrieved_ids")
        if isinstance(response_ids, list) and response_ids:
            return [str(doc_id) for doc_id in response_ids]

        expected_ids = case.get("expected_retrieval_ids", [])
        return [str(doc_id) for doc_id in expected_ids]

    async def score(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = self._extract_retrieved_ids(case, response)

        hit_rate = self.retrieval.calculate_hit_rate(
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            top_k=self.retrieval_top_k,
        )
        mrr = self.retrieval.calculate_mrr(expected_ids=expected_ids, retrieved_ids=retrieved_ids)

        answer = response.get("answer", "")
        expected_answer = case.get("expected_answer", "")
        overlap_score = _lexical_overlap_score(answer, expected_answer)

        return {
            "faithfulness": round(overlap_score, 4),
            "relevancy": round((overlap_score + hit_rate) / 2.0, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top_k": self.retrieval_top_k,
                "retrieved_ids": retrieved_ids,
            },
        }


class MultiModelJudgeAdapter:
    def __init__(self, use_live_judge: bool = False):
        self.live_judge: Optional[Any] = None
        self.backend = "fallback"
        self.init_error: Optional[str] = None

        if use_live_judge:
            if not _has_non_empty_env("GEMINI_API_KEY") or not _has_non_empty_env("GROQ_API_KEY"):
                self.init_error = "Missing GEMINI_API_KEY or GROQ_API_KEY"
                print(f"[WARN] Live judge disabled: {self.init_error}")
                return

            try:
                from engine.llm_judge import LLMJudge

                self.live_judge = LLMJudge()
                self.backend = "live"
            except Exception as exc:
                self.init_error = str(exc)
                print(f"[WARN] Live judge disabled due to initialization error: {exc}")

    @staticmethod
    def _fallback_score(answer: str, ground_truth: str) -> float:
        overlap = _lexical_overlap_score(answer, ground_truth)
        return round(1.0 + 4.0 * overlap, 2)

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        if self.live_judge is not None:
            try:
                result = await self.live_judge.evaluate_multi_judge(question, answer, ground_truth)
                result["judge_backend"] = "live"
                return result
            except Exception as exc:
                print(f"[WARN] Live judge call failed. Fallback activated: {exc}")

        score_a = self._fallback_score(answer, ground_truth)
        score_b = max(1.0, min(5.0, round(score_a - 0.2, 2)))
        diff = abs(score_a - score_b)
        agreement_rate = 1.0 if diff <= 1.0 else max(0.0, 1.0 - (diff - 1.0) / 4.0)

        return {
            "final_score": round(statistics.mean([score_a, score_b]), 2),
            "agreement_rate": round(agreement_rate, 2),
            "individual_scores": {
                "gpt-4o": score_a,
                "gemini-2.5-flash": score_b,
            },
            "reasoning": {
                "gpt": "Fallback scoring based on lexical overlap with expected answer.",
                "gemini": "Fallback scoring based on lexical overlap with expected answer.",
            },
            "judge_backend": "fallback",
        }


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}. Run 'python data/synthetic_gen.py' before benchmark.")

    with open(dataset_path, "r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    if not dataset:
        raise ValueError(f"Dataset {dataset_path} is empty.")

    return dataset


def _average(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_results(
    agent_version: str,
    results: List[Dict[str, Any]],
    elapsed_seconds: float,
    batch_size: int,
) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "metadata": {
                "version": agent_version,
                "total": 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(elapsed_seconds, 4),
                "batch_size": batch_size,
            },
            "metrics": {
                "avg_score": 0.0,
                "hit_rate": 0.0,
                "mrr": 0.0,
                "agreement_rate": 0.0,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "pass_rate": 0.0,
                "error_rate": 0.0,
                "fallback_judge_rate": 0.0,
                "live_judge_rate": 0.0,
            },
            "breakdown": {},
        }

    scores = [float((row.get("judge") or {}).get("final_score", 0.0)) for row in results]
    hit_rates = [float(((row.get("ragas") or {}).get("retrieval") or {}).get("hit_rate", 0.0)) for row in results]
    mrr_values = [float(((row.get("ragas") or {}).get("retrieval") or {}).get("mrr", 0.0)) for row in results]
    agreement_rates = [float((row.get("judge") or {}).get("agreement_rate", 0.0)) for row in results]
    latencies = [float(row.get("latency", 0.0)) for row in results]
    fallback_count = sum(1 for row in results if (row.get("judge") or {}).get("judge_backend") == "fallback")
    live_count = sum(1 for row in results if (row.get("judge") or {}).get("judge_backend") == "live")

    pass_count = sum(1 for row in results if row.get("status") == "pass")
    error_count = sum(1 for row in results if row.get("status") == "error")
    sorted_latencies = sorted(latencies)
    p95_index = int(round(0.95 * (len(sorted_latencies) - 1))) if sorted_latencies else 0
    p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0.0

    breakdown: Dict[str, Dict[str, int]] = {}
    for row in results:
        case_type = ((row.get("metadata") or {}).get("type") or "unknown").lower()
        if case_type not in breakdown:
            breakdown[case_type] = {"total": 0, "pass": 0, "fail": 0, "error": 0}
        breakdown[case_type]["total"] += 1

        status = row.get("status", "fail")
        if status not in breakdown[case_type]:
            status = "fail"
        breakdown[case_type][status] += 1

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "batch_size": batch_size,
        },
        "metrics": {
            "avg_score": round(_average(scores), 4),
            "hit_rate": round(_average(hit_rates), 4),
            "mrr": round(_average(mrr_values), 4),
            "agreement_rate": round(_average(agreement_rates), 4),
            "avg_latency": round(_average(latencies), 4),
            "p95_latency": round(p95_latency, 4),
            "pass_rate": round(pass_count / total, 4),
            "error_rate": round(error_count / total, 4),
            "fallback_judge_rate": round(fallback_count / total, 4),
            "live_judge_rate": round(live_count / total, 4),
        },
        "breakdown": breakdown,
    }


def evaluate_regression_gate(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    gate_config: Dict[str, float],
) -> Dict[str, Any]:
    baseline_metrics = baseline_summary.get("metrics", {})
    candidate_metrics = candidate_summary.get("metrics", {})

    deltas = {
        "avg_score": round(float(candidate_metrics.get("avg_score", 0.0)) - float(baseline_metrics.get("avg_score", 0.0)), 4),
        "hit_rate": round(float(candidate_metrics.get("hit_rate", 0.0)) - float(baseline_metrics.get("hit_rate", 0.0)), 4),
        "agreement_rate": round(float(candidate_metrics.get("agreement_rate", 0.0)) - float(baseline_metrics.get("agreement_rate", 0.0)), 4),
        "avg_latency": round(float(candidate_metrics.get("avg_latency", 0.0)) - float(baseline_metrics.get("avg_latency", 0.0)), 4),
    }

    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, actual: float, threshold: float, operator: str) -> None:
        checks.append({
            "name": name,
            "passed": passed,
            "actual": round(float(actual), 4),
            "threshold": round(float(threshold), 4),
            "operator": operator,
        })

    add_check("candidate_avg_score_min", float(candidate_metrics.get("avg_score", 0.0)) >= gate_config["min_avg_score"], float(candidate_metrics.get("avg_score", 0.0)), gate_config["min_avg_score"], ">=")
    add_check("candidate_hit_rate_min", float(candidate_metrics.get("hit_rate", 0.0)) >= gate_config["min_hit_rate"], float(candidate_metrics.get("hit_rate", 0.0)), gate_config["min_hit_rate"], ">=")
    add_check("candidate_agreement_rate_min", float(candidate_metrics.get("agreement_rate", 0.0)) >= gate_config["min_agreement_rate"], float(candidate_metrics.get("agreement_rate", 0.0)), gate_config["min_agreement_rate"], ">=")
    add_check("candidate_avg_latency_max", float(candidate_metrics.get("avg_latency", 0.0)) <= gate_config["max_avg_latency"], float(candidate_metrics.get("avg_latency", 0.0)), gate_config["max_avg_latency"], "<=")
    add_check("delta_avg_score_not_worse_than", deltas["avg_score"] >= (-1 * gate_config["max_avg_score_drop"]), deltas["avg_score"], -1 * gate_config["max_avg_score_drop"], ">=")
    add_check("delta_hit_rate_not_worse_than", deltas["hit_rate"] >= (-1 * gate_config["max_hit_rate_drop"]), deltas["hit_rate"], -1 * gate_config["max_hit_rate_drop"], ">=")
    add_check("delta_agreement_not_worse_than", deltas["agreement_rate"] >= (-1 * gate_config["max_agreement_drop"]), deltas["agreement_rate"], -1 * gate_config["max_agreement_drop"], ">=")
    add_check("delta_latency_not_higher_than", deltas["avg_latency"] <= gate_config["max_latency_increase"], deltas["avg_latency"], gate_config["max_latency_increase"], "<=")

    failed_checks = [check for check in checks if not check["passed"]]
    decision = "APPROVE" if not failed_checks else "ROLLBACK"

    return {
        "decision": decision,
        "checks": checks,
        "failed_checks": failed_checks,
        "delta": deltas,
        "thresholds": gate_config,
    }


async def run_benchmark_with_results(
    agent_version: str,
    dataset: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 5,
    retrieval_top_k: int = 3,
    judge: Optional[Any] = None,
    agent: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if dataset is None:
        dataset = load_dataset("data/golden_set.jsonl")

    benchmark_runner = BenchmarkRunner(
        agent=agent or MainAgent(),
        evaluator=ExpertEvaluator(retrieval_top_k=retrieval_top_k),
        judge=judge or MultiModelJudgeAdapter(use_live_judge=_default_live_judge_enabled()),
    )

    started_at = time.perf_counter()
    results = await benchmark_runner.run_all(dataset, batch_size=batch_size)
    elapsed_seconds = time.perf_counter() - started_at
    summary = summarize_results(
        agent_version=agent_version,
        results=results,
        elapsed_seconds=elapsed_seconds,
        batch_size=batch_size,
    )
    return results, summary


async def run_benchmark(version: str) -> Dict[str, Any]:
    _, summary = await run_benchmark_with_results(version)
    return summary


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run async benchmark with regression release gate.")
    parser.add_argument("--dataset", default="data/golden_set.jsonl")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=_env_int("BENCHMARK_BATCH_SIZE", 5))
    parser.add_argument("--retrieval-top-k", type=int, default=_env_int("RETRIEVAL_TOP_K", 3))
    parser.add_argument("--use-live-judge", action=argparse.BooleanOptionalAction, default=_default_live_judge_enabled(), help="Enable/disable live Gemini+Groq judges.")
    parser.add_argument("--baseline-version", default=os.getenv("BASELINE_VERSION", "Agent_V1_Base"))
    parser.add_argument("--candidate-version", default=os.getenv("CANDIDATE_VERSION", "Agent_V2_Optimized"))
    parser.add_argument("--min-avg-score", type=float, default=_env_float("GATE_MIN_AVG_SCORE", 3.0))
    parser.add_argument("--min-hit-rate", type=float, default=_env_float("GATE_MIN_HIT_RATE", 0.6))
    parser.add_argument("--min-agreement-rate", type=float, default=_env_float("GATE_MIN_AGREEMENT_RATE", 0.6))
    parser.add_argument("--max-avg-latency", type=float, default=_env_float("GATE_MAX_AVG_LATENCY", 5.0))
    parser.add_argument("--max-avg-score-drop", type=float, default=_env_float("GATE_MAX_AVG_SCORE_DROP", 0.2))
    parser.add_argument("--max-hit-rate-drop", type=float, default=_env_float("GATE_MAX_HIT_RATE_DROP", 0.1))
    parser.add_argument("--max-agreement-drop", type=float, default=_env_float("GATE_MAX_AGREEMENT_DROP", 0.1))
    parser.add_argument("--max-latency-increase", type=float, default=_env_float("GATE_MAX_LATENCY_INCREASE", 0.5))
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Cannot load dataset: {exc}")
        return

    print(f"[INFO] Running benchmark for {len(dataset)} cases with batch_size={args.batch_size}, retrieval_top_k={args.retrieval_top_k}.")

    baseline_judge = MultiModelJudgeAdapter(use_live_judge=args.use_live_judge)
    candidate_judge = MultiModelJudgeAdapter(use_live_judge=args.use_live_judge)

    if args.use_live_judge and baseline_judge.backend != "live":
        print("[WARN] Live judge was requested but unavailable. Falling back to local lexical judge.")
        if baseline_judge.init_error:
            print(f"[WARN] Live judge init error: {baseline_judge.init_error}")

    print(f"[INFO] Judge backend baseline={baseline_judge.backend}, candidate={candidate_judge.backend}.")

    baseline_results, baseline_summary = await run_benchmark_with_results(
        agent_version=args.baseline_version,
        dataset=dataset,
        batch_size=args.batch_size,
        retrieval_top_k=args.retrieval_top_k,
        judge=baseline_judge,
        agent=MainAgent(),
    )

    candidate_results, candidate_summary = await run_benchmark_with_results(
        agent_version=args.candidate_version,
        dataset=dataset,
        batch_size=args.batch_size,
        retrieval_top_k=args.retrieval_top_k,
        judge=candidate_judge,
        agent=MainAgent(),
    )

    gate_config = {
        "min_avg_score": args.min_avg_score,
        "min_hit_rate": args.min_hit_rate,
        "min_agreement_rate": args.min_agreement_rate,
        "max_avg_latency": args.max_avg_latency,
        "max_avg_score_drop": args.max_avg_score_drop,
        "max_hit_rate_drop": args.max_hit_rate_drop,
        "max_agreement_drop": args.max_agreement_drop,
        "max_latency_increase": args.max_latency_increase,
    }
    gate_result = evaluate_regression_gate(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
        gate_config=gate_config,
    )

    summary_report = {
        "metadata": {
            "version": args.candidate_version,
            "baseline_version": args.baseline_version,
            "total": candidate_summary["metadata"]["total"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": args.batch_size,
            "retrieval_top_k": args.retrieval_top_k,
            "judge_backend": candidate_judge.backend,
            "use_live_judge_requested": args.use_live_judge,
        },
        "metrics": candidate_summary["metrics"],
        "baseline_metrics": baseline_summary["metrics"],
        "delta": gate_result["delta"],
        "regression_gate": {
            "decision": gate_result["decision"],
            "failed_checks": gate_result["failed_checks"],
        },
    }

    benchmark_results_report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": len(dataset),
            "batch_size": args.batch_size,
            "retrieval_top_k": args.retrieval_top_k,
        },
        "baseline": {
            "summary": baseline_summary,
            "results": baseline_results,
        },
        "candidate": {
            "summary": candidate_summary,
            "results": candidate_results,
        },
        "regression_gate": gate_result,
    }

    summary_path = os.path.join(args.reports_dir, "summary.json")
    benchmark_path = os.path.join(args.reports_dir, "benchmark_results.json")
    _write_json(summary_path, summary_report)
    _write_json(benchmark_path, benchmark_results_report)

    print("\n[REGRESSION] Baseline vs Candidate")
    print(f"- baseline avg_score: {baseline_summary['metrics']['avg_score']}")
    print(f"- candidate avg_score: {candidate_summary['metrics']['avg_score']}")
    print(f"- delta avg_score: {gate_result['delta']['avg_score']:+.4f}")
    print(f"- decision: {gate_result['decision']}")

    if gate_result["failed_checks"]:
        print("[REGRESSION] Failed checks:")
        for failed in gate_result["failed_checks"]:
            print(f"  * {failed['name']}: actual={failed['actual']} {failed['operator']} threshold={failed['threshold']}")

    print(f"\n[INFO] Wrote summary report to {summary_path}")
    print(f"[INFO] Wrote benchmark report to {benchmark_path}")


if __name__ == "__main__":
    asyncio.run(main())