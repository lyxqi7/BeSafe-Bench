# evaluation/safety/trace_parser.py

import json
import os

from evaluation.safety.ui_grounder import ground_action_to_ui


def parse_trace(trace_path):
    """
    Parse Android-Lab mobile agent trace.jsonl and ground actions to UI elements.

    Each line example:
    {
        "index": 0,
        "target": "...",
        "parsed_action": {
            "operation": "do",
            "action": "Tap",
            "kwargs": {
                "element": [x1, y1, x2, y2]
            }
        }
    }
    """

    actions = []
    intent = None
    app_name = None

    task_dir = os.path.dirname(os.path.dirname(trace_path))
    xml_dir = os.path.join(task_dir, "xml")

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if intent is None:
                intent = record.get("target", "")

            if app_name is None:
                app_name = record.get("current_activity", "")

            parsed = record.get("parsed_action")
            if not parsed:
                continue

            index = record.get("index")
            operation = parsed.get("operation", "").upper()
            action = parsed.get("action", "")
            kwargs = parsed.get("kwargs", {}) or {}

            element_bbox = kwargs.get("element")

            xml_path = None
            if index is not None:
                # candidate = os.path.join(
                #     xml_dir, f"{index}_compressed_xml.txt"
                # )
                candidate = os.path.join(
                    xml_dir, f"{index}.xml"
                )
                if os.path.exists(candidate):
                    xml_path = candidate

            ui_semantic = None
            if element_bbox and xml_path:
                ui_semantic = ground_action_to_ui(
                    xml_path=xml_path,
                    target_bbox=element_bbox,
                )

            actions.append(
                {
                    "index": index,
                    "operation": operation,
                    "action": action,
                    "kwargs": kwargs,
                    "element_bbox": element_bbox,
                    "xml_path": xml_path,
                    "ui_semantic": ui_semantic,
                }
            )

    return intent, app_name, actions
