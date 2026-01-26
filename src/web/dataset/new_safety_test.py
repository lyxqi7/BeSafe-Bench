import os
import json

total_json_path = ""
single_json_folder = ""
output_json_path = ""

with open(total_json_path, "r") as f:
    total_data = json.load(f)

for i, element in enumerate(total_data):
    single_file_path = os.path.join(single_json_folder, f"{i}.json")
    if os.path.exists(single_file_path):
        with open(single_file_path, "r") as f:
            single_data = json.load(f)
        element["intent_template"] = single_data.get("intent_template", element.get("intent_template"))
        element["instantiation_dict"] = single_data.get("instantiation_dict", element.get("instantiation_dict"))
        element["intent"] = single_data.get("intent", element.get("intent"))
        element["risk_type"] = single_data.get("risk_type", element.get("risk_type"))
        element["risk_trigger_mechanism"] = single_data.get("risk_trigger_mechanism", element.get("risk_trigger_mechanism"))
        if "reference_answers" in single_data:
            element["reference_answers"] = single_data["reference_answers"]
        if "eval" in element and "reference_answers" in element["eval"]:
            element["eval"]["reference_answers"] = single_data.get("reference_answers", element["eval"]["reference_answers"])
    else:
        print(f"loss: {single_file_path}")

with open(output_json_path, "w") as f:
    json.dump(total_data, f, indent=2)

print("finish total_updated.json")
