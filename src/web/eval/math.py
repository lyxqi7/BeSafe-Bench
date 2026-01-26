import os
import json
import re
from collections import Counter

# 配置路径
LOG_ROOT = os.path.join("log", "AWM")
CONFIG_ROOT = r"C:\Users\qi\Desktop\huawei\agent\web\agent-workflow-memory\webarena\config_files"


def iter_outputs_jsons(root):
    """
    遍历获取所有的 outputs.jsons 文件路径
    """
    for dirpath, dirnames, filenames in os.walk(root):
        if "outputs.jsons" in filenames and dirpath.endswith("trajs"):
            yield os.path.join(dirpath, "outputs.jsons")


def get_config_id(path):
    """
    从路径中提取 webarena. 后的数字
    例如: ...webarena.21_9... -> 21
    """
    # 使用正则匹配 webarena. 后面紧跟的数字
    match = re.search(r'webarena\.(\d+)', path)
    return match.group(1) if match else None


def should_skip_by_config(config_id):
    """
    检查配置文件中是否包含 gitlab
    """
    if not config_id:
        return False

    config_path = os.path.join(CONFIG_ROOT, f"{config_id}.json")

    if not os.path.exists(config_path):
        return False  # 如果配置文件不存在，默认不跳过（或者你可以根据需求改为True）

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 检查 sites 列表中是否包含 gitlab
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
        # --- 新增逻辑：提取ID并过滤 gitlab ---
        config_id = get_config_id(path)
        if should_skip_by_config(config_id):
            skipped_gitlab += 1
            continue
        # ------------------------------------

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