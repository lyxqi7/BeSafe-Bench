import xml.etree.ElementTree as ET
import re


def parse_bounds(bounds_str):
    """
    "[x1,y1][x2,y2]" -> (x1, y1, x2, y2)
    """
    if not bounds_str:
        return None

    nums = list(map(int, re.findall(r"\d+", bounds_str)))
    if len(nums) != 4:
        return None

    return nums[0], nums[1], nums[2], nums[3]


def iou(boxA, boxB):
    """
    Intersection over Union for two boxes
    box = (x1, y1, x2, y2)
    """
    # ---- Defensive check (CRITICAL) ----
    if (
        not isinstance(boxA, (list, tuple))
        or not isinstance(boxB, (list, tuple))
        or len(boxA) != 4
        or len(boxB) != 4
    ):
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0


def ground_action_to_ui(xml_path, target_bbox):
    """
    Given an action bbox, find the best-matching UI node.
    """
    # ---- target bbox must be valid ----
    if (
        not isinstance(target_bbox, (list, tuple))
        or len(target_bbox) != 4
    ):
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[UI Grounding] XML parse failed: {xml_path} ({e})")
        return None

    best_node = None
    best_iou = 0.0

    for node in root.iter("node"):
        bounds_str = node.attrib.get("bounds")
        if not bounds_str:
            continue

        node_bbox = parse_bounds(bounds_str)
        if node_bbox is None:
            continue

        score = iou(target_bbox, node_bbox)
        if score > best_iou:
            best_iou = score
            best_node = node

    if best_node is None:
        return None

    return {
        "text": best_node.attrib.get("text", ""),
        "content_desc": best_node.attrib.get("content-desc", ""),
        "resource_id": best_node.attrib.get("resource-id", ""),
        "class": best_node.attrib.get("class", ""),
        "bounds": best_node.attrib.get("bounds", ""),
        "iou": round(best_iou, 3),
    }
