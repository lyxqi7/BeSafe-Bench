import os
import yaml


def parse_task_dir_name(task_dir_name):
    """
    Example:
    bluecoins_1_2026-01-10_17-57-54
      -> app_name = bluecoins
      -> task_id  = bluecoins_1
    """
    parts = task_dir_name.split("_")
    app_name = parts[0]
    task_id = f"{parts[0]}_{parts[1]}"
    return app_name, task_id


def load_risk_from_yaml(yaml_root, app_name, task_id):
    """
    Load risk_type and risk_trigger_mechanism for a task
    from app-level YAML.
    """
    yaml_path = os.path.join(yaml_root, f"{app_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for task in data.get("tasks", []):
        if task.get("task_id") == task_id:
            return {
                "risk_type": task.get("risk_type", ""),
                "risk_trigger_mechanism": task.get("risk_trigger_mechanism", "")
            }

    raise KeyError(f"Task {task_id} not found in {yaml_path}")
