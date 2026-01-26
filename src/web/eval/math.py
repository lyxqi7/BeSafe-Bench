import os
import json
import re
from collections import Counter

LOG_ROOT = os.path.join("log", "AWM")
CONFIG_ROOT = r""


def iter_outputs_jsons(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "outputs.jsons" in filenames and dirpath.endswith("trajs"):
            yield os.path.join(dirpath, "outputs.jsons")


def get_config_id(path):
    match = re.search(r'webarena\.(\d+)', path)
    return match.group(1) if match else None


def should_skip_by_config(config_id):
    if not config_id:
        return False

    config_path = os.path.join(CONFIG_ROOT, f"{config_id}.json")

    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if 'gitlab' in data.get('sites', []):
                return True
    except Exception as e:
        print(f"Warning: Could not read config {config_path}: {e}")

    return False


def main():
    total = 0
    task_counter = Counter()
    safety_counter = Counter()
    joint_counter = Counter()

    malformed = 0
    skipped_gitlab = 0

    for path in iter_outputs_jsons(LOG_ROOT):
        config_id = get_config_id(path)
        if should_skip_by_config(config_id):
            skipped_gitlab += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    eval_info = record.get("eval_info", {})

                    task_status = eval_info.get("task", {}).get("status", "unknown")
                    safety_status = eval_info.get("safety", {}).get("status", "unknown")

                    task_counter[task_status] += 1
                    safety_counter[safety_status] += 1
                    joint_counter[(task_status, safety_status)] += 1

                    total += 1

                except Exception:
                    malformed += 1

    print("========== Evaluation Statistics ==========")
    print(f"Total trajectories (Filtered) : {total}")
    print(f"Skipped (GitLab)             : {skipped_gitlab}")
    print(f"Malformed entries            : {malformed}")
    print()

    if total == 0:
        print("No data to display.")
        return

    print("---- Task Performance ----")
    for k, v in task_counter.items():
        print(f"{k:>10}: {v:5d} ({v / total:.2%})")
    print()

    print("---- Safety Assessment ----")
    for k, v in safety_counter.items():
        print(f"{k:>10}: {v:5d} ({v / total:.2%})")
    print()

    print("---- Joint Distribution (Task × Safety) ----")
    for (task_s, safety_s), v in joint_counter.items():
        print(
            f"Task={task_s:<8} | Safety={safety_s:<8} : "
            f"{v:5d} ({v / total:.2%})"
        )

    print("\n===========================================")


if __name__ == "__main__":
    main()