import os
import json
import argparse
import re
import subprocess
import sys
import datetime


def main():
    # 1. 强化环境变量设置
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"  # 强制子进程不使用缓冲区，实时打印日志

    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    config_list = [json.load(open(f)) for f in config_files]
    config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]

    if args.end_index == None: args.end_index = len(task_ids)

    # 增加时间戳防止日志覆盖
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_file = f"error_log_{args.website}_{timestamp}.txt"

    with open(log_file, "a", encoding="utf-8") as log:
        for tid in task_ids[args.start_index: args.end_index]:
            print(f"\n" + "=" * 50)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 正在处理任务: webarena.{tid}")
            print("=" * 50)

            # --- Step 1: Run Inference ---
            # 使用 python -u 确保输出即时刷新
            cmd_step1 = [
                "python", "-u", "run.py",
                "--task", f"webarena.{tid}",
                "--workflow_path", f"workflow/{args.website}.txt"
            ]
            try:
                print(f"执行 Step 1: {' '.join(cmd_step1)}")
                # 设置超时时间为 600 秒 (10分钟)，防止卡死
                subprocess.run(cmd_step1, check=True, env=env, timeout=600)

            except subprocess.TimeoutExpired:
                error_msg = f"[Task {tid}] ❌ 严重错误: Step 1 运行超时 (10分钟)，已跳过该任务。\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()
                continue  # 超时直接跳过此任务

            except subprocess.CalledProcessError as e:
                error_msg = f"[Task {tid}] ❌ Step 1 (Inference) 失败: {e}\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()
                continue

                # --- Step 2: Run Evaluation ---
            evaluation_path = find_webarena_dirname("results", tid)
            if not evaluation_path:
                error_msg = f"[Task {tid}] ⚠️ Step 2 跳过: 找不到结果目录，可能推理未成功结束。\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()
            else:
                cmd_step2 = [
                    "python", "-u", "-m", "autoeval.evaluate_trajectory",
                    "--result_dir", f"results/{evaluation_path}"
                ]
                try:
                    print(f"执行 Step 2: {' '.join(cmd_step2)}")
                    # 评测脚本通常较快，设置 3 分钟超时
                    subprocess.run(cmd_step2, check=True, env=env, timeout=600)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    error_msg = f"[Task {tid}] ⚠️ Step 2 (Evaluation) 失败或超时: {e}\n"
                    print(error_msg)
                    log.write(error_msg)
                    log.flush()

            # --- Step 3: Update Workflow ---
            cmd_step3 = [
                "python", "-u", "induce_prompt.py",
                "--result_dir", "results",
                "--output_path", f"workflow/{args.website}.txt",
            ]
            try:
                print(f"执行 Step 3: {' '.join(cmd_step3)}")
                subprocess.run(cmd_step3, check=True, env=env, timeout=180)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                error_msg = f"[Task {tid}] ⚠️ Step 3 (Update Workflow) 失败或超时: {e}\n"
                print(error_msg)
                log.write(error_msg)
                log.flush()

    print(f"\n所有任务处理完毕。错误详情请查看: {log_file}")


def find_webarena_dirname(base_dir, target_id, year=2026):
    if not os.path.exists(base_dir):
        return None
    target_id_str = f"webarena.{target_id}"
    pattern = re.compile(rf"^{year}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}}_.*_{target_id_str}(_\d+)?$")

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and pattern.match(entry):
            return entry
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()

    main()