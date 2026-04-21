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
from agent.main_agent import MainAgent

# Giả lập các components Expert
class ExpertEvaluator:
    async def score(self, case, resp): 
        # Giả lập tính toán Hit Rate và MRR
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }


class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt): 
        return {
            "final_score": 4.5, 
            "agreement_rate": 0.8,
            "reasoning": "Cả 2 model đồng ý đây là câu trả lời tốt."
        }

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

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
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print(
        f"[INFO] Running benchmark for {len(dataset)} cases with batch_size={args.batch_size}, "
        f"retrieval_top_k={args.retrieval_top_k}."
    )

    baseline_judge = MultiModelJudgeAdapter(use_live_judge=args.use_live_judge)
    candidate_judge = MultiModelJudgeAdapter(use_live_judge=args.use_live_judge)

    if args.use_live_judge and baseline_judge.backend != "live":
        print("[WARN] Live judge was requested but unavailable. Falling back to local lexical judge.")
        if baseline_judge.init_error:
            print(f"[WARN] Live judge init error: {baseline_judge.init_error}")

    print(
        f"[INFO] Judge backend baseline={baseline_judge.backend}, candidate={candidate_judge.backend}."
    )

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
            print(
                f"  * {failed['name']}: actual={failed['actual']} "
                f"{failed['operator']} threshold={failed['threshold']}"
            )

    print(f"\n[INFO] Wrote summary report to {summary_path}")
    print(f"[INFO] Wrote benchmark report to {benchmark_path}")



if __name__ == "__main__":
    asyncio.run(main())
