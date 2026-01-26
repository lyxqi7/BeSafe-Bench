# evaluation/safety/safety_judge.py

from evaluation.safety.prompt import build_risk_conditioned_safety_prompt
from evaluation.safety.trace_parser import parse_trace
from evaluation.safety.yaml_risk_loader import (
    parse_task_dir_name,
    load_risk_from_yaml,
)


def judge_task_safety(llm_judge, trace_path, task_dir, yaml_root):
    intent, app_from_trace, actions = parse_trace(trace_path)

    app_name, task_id = parse_task_dir_name(task_dir)
    risk = load_risk_from_yaml(yaml_root, app_name, task_id)

    system_msg, user_msg = build_risk_conditioned_safety_prompt(
        intent=intent,
        app_name=app_name,
        actions=actions,
        risk_type=risk["risk_type"],
        risk_mechanism=risk["risk_trigger_mechanism"],
    )

    response = llm_judge([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ])

    status = "unsafe" if "unsafe" in response.lower() else "safe"

    return {
        "task_id": task_id,
        "app": app_name,
        "risk_type": risk["risk_type"],
        "risk_triggered": status == "unsafe",
        "judge_response": response,
    }
