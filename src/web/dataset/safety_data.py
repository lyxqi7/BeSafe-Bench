import json
import os

base_path = "safety_benchmark"

required_keys = {"intent_template", "instantiation_dict", "intent", "reference_answers", "risk_type", "risk_trigger_mechanism"}

for i in range(812):
    filename = os.path.join(base_path, f"{i}.json")

    if not os.path.isfile(filename):
        print(f"{filename}")
        continue

    try:
        with open(filename, "r") as f:
            raw = f.read().strip()

        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]

        try:
            raw2 = json.loads(f'"{raw}"')
            obj = json.loads(raw2)
        except Exception:
            print(f"{filename}")
            continue

        if not isinstance(obj, dict):
            print(f"{filename} dict")
            continue

        if not required_keys.issubset(obj.keys()):
            print(f"{filename} key")
            continue

        with open(filename, "w") as f:
            json.dump(obj, f, indent=2)

    except Exception as e:
        print(f"{filename}: {e}")
