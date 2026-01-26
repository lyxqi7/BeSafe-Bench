import os
import yaml
from copy import deepcopy

INPUT_DIR = "yaml_output"
OUTPUT_DIR = "yaml_stripped"

TASK_KEYS_TO_KEEP = {
    "task_id",
    "task",
    "metric_type",
    "metric_func",
}

def process_yaml_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML is not a dict: {input_path}")

    new_data = deepcopy(data)

    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"'tasks' is not a list in {input_path}")

    new_tasks = []
    for task in tasks:
        if not isinstance(task, dict):
            continue

        stripped_task = {
            k: task[k]
            for k in TASK_KEYS_TO_KEEP
            if k in task
        }

        new_tasks.append(stripped_task)

    new_data["tasks"] = new_tasks

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            new_data,
            f,
            allow_unicode=True,
            sort_keys=False
        )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if not (filename.endswith(".yaml") or filename.endswith(".yml")):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            process_yaml_file(input_path, output_path)
            print(f"[OK] Processed: {filename}")
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")

if __name__ == "__main__":
    main()
