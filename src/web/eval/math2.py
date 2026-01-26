import os
import json
import re
from collections import defaultdict

LOG_ROOT = os.path.join("log", "AWM")
CONFIG_ROOT = r""


def iter_outputs_jsons(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "outputs.jsons" in filenames and dirpath.endswith("trajs"):
            yield os.path.join(dirpath, "outputs.jsons")


def get_config_info(path):
    match = re.search(r'webarena\.(\d+)', path)
    if not match:
        return None, None, None

    config_id = match.group(1)
    config_path = os.path.join(CONFIG_ROOT, f"{config_id}.json")

    if not os.path.exists(config_path):
        return config_id, None, None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sites = data.get('sites', [])
            risk_type = data.get('risk_type', 'unknown')
            return config_id, sites, risk_type
    except Exception:
        return config_id, None, None


def print_table(title, data_dict, category_name):
    print(f"\n---- {title} ----")
    header = f"{category_name:<20} | {'Task Success':<15} | {'Safety Pass':<15} | {'Total'}"
    print(header)
    print("-" * len(header))

    for key in sorted(data_dict.keys()):
        stats = data_dict[key]
        t_success = stats['task'].get('success', 0)
        s_success = stats['safety'].get('safe', 0)

        t_total = sum(stats['task'].values())

        t_rate = (t_success / t_total * 100) if t_total > 0 else 0
        s_rate = (s_success / t_total * 100) if t_total > 0 else 0

        print(
            f"{str(key):<20} | {t_success:>4}/{t_total:<4} ({t_rate:>5.1f}%) | {s_success:>4}/{t_total:<4} ({s_rate:>5.1f}%) | {t_total}")


def main():
    analysis = {
        'site': defaultdict(lambda: {'task': defaultdict(int), 'safety': defaultdict(int)}),
        'risk_type': defaultdict(lambda: {'task': defaultdict(int), 'safety': defaultdict(int)})
    }

    total_count = 0
    skipped_gitlab = 0

    for path in iter_outputs_jsons(LOG_ROOT):
        config_id, sites, risk_type = get_config_info(path)

        if sites and 'gitlab' in sites:
            skipped_gitlab += 1
            continue

        if not sites: sites = ['unknown_site']

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    eval_info = record.get("eval_info", {})

                    t_status = eval_info.get("task", {}).get("status", "unknown")
                    s_status = eval_info.get("safety", {}).get("status", "unknown")

                    for site in sites:
                        analysis['site'][site]['task'][t_status] += 1
                        analysis['site'][site]['safety'][s_status] += 1

                    analysis['risk_type'][risk_type]['task'][t_status] += 1
                    analysis['risk_type'][risk_type]['safety'][s_status] += 1

                    total_count += 1
                except Exception:
                    continue

    print(f"Total processed: {total_count} (Skipped GitLab: {skipped_gitlab})")

    print_table("Performance by Site", analysis['site'], "Site Name")
    print_table("Performance by Risk Type", analysis['risk_type'], "Risk Type")


if __name__ == "__main__":
    main()