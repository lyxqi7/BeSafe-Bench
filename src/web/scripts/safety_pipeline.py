import json
import subprocess


log_file = "command_errors.log"
# E1 = 189
# e2 = 316
# e3 = 317

total_json_path = "total_safety.json"
with open(total_json_path, "r") as f:
    total_data = json.load(f)

with open(log_file, "a+") as log:
    for i in range(812):
        if i < 317:
            continue
        if 'gitlab' in total_data[i]['sites']:
            log.write(f'{i} is reddit')
            log.flush()
            continue
        cmd = ["python", "run.py", "--task_name", f"webarena.{i}"]
        try:
            print(f"{' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            error_msg = f"webarena.{i} : {e}\n"
            print(error_msg)
            log.write(error_msg)
            log.flush()

print(f"{log_file}")
