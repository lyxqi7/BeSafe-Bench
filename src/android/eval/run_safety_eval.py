# evaluation/safety/run_safety_eval.py

import os
import json
import traceback
from evaluation.safety.safety_judge import judge_task_safety


def run_safety_eval(agent_log_dir, yaml_root, llm_judge):
    total = 0
    triggered = 0
    per_risk = {}

    for task_dir in os.listdir(agent_log_dir):
        trace_path = os.path.join(
            agent_log_dir, task_dir, "traces", "trace.jsonl"
        )

        if not os.path.exists(trace_path):
            continue

        try:
            result = judge_task_safety(
                llm_judge=llm_judge,
                trace_path=trace_path,
                task_dir=task_dir,
                yaml_root=yaml_root,
            )
        except Exception as e:
            print(f"[ERROR] Safety evaluation failed for task: {task_dir}")
            print(f"[ERROR] {repr(e)}")
            traceback.print_exc()
            continue

        # 只有成功的 task 才计入统计
        total += 1

        risk_type = result["risk_type"]
        per_risk.setdefault(risk_type, {"total": 0, "triggered": 0})
        per_risk[risk_type]["total"] += 1

        if result["risk_triggered"]:
            triggered += 1
            per_risk[risk_type]["triggered"] += 1

        with open(
            os.path.join(agent_log_dir, task_dir, "safety.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Finish {task_dir}")

    summary = {
        "total_tasks": total,
        "risk_triggered_tasks": triggered,
        "safety_coverage": 1 - triggered / max(total, 1),
        "per_risk": {
            k: {
                "total": v["total"],
                "triggered": v["triggered"],
                "coverage": 1 - v["triggered"] / max(v["total"], 1),
            }
            for k, v in per_risk.items()
        },
    }

    with open(
        os.path.join(agent_log_dir, "safety_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
