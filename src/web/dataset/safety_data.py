import json
import os

base_path = "safety_benchmark"   # 如果文件就在当前目录，不需要修改；否则改为你的路径

required_keys = {"intent_template", "instantiation_dict", "intent", "reference_answers", "risk_type", "risk_trigger_mechanism"}

for i in range(812):
    filename = os.path.join(base_path, f"{i}.json")

    if not os.path.isfile(filename):
        print(f"文件不存在: {filename}")
        continue

    try:
        with open(filename, "r") as f:
            raw = f.read().strip()

        # Step 1: 去掉外层引号
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]

        # Step 2: 将内部转义字符恢复
        try:
            # 内层内容还是 JSON 格式，但有转义符，所以再 loads 一次
            raw2 = json.loads(f'"{raw}"')
            obj = json.loads(raw2)
        except Exception:
            print(f"{filename} 内部 JSON 解析失败")
            continue

        # Step 3：必须是 dict
        if not isinstance(obj, dict):
            print(f"{filename} 内部不是 dict")
            continue

        # Step 4：检查 key 是否都存在
        if not required_keys.issubset(obj.keys()):
            print(f"{filename} 缺少关键 key")
            continue

        # Step 5：写回文件（写成真正 JSON）
        with open(filename, "w") as f:
            json.dump(obj, f, indent=2)

    except Exception as e:
        print(f"{filename} 处理出错: {e}")
