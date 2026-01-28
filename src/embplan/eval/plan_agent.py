import json
import os   
import re
import sys
from typing import Generator, List, Tuple, Optional

from og_ego_prim.models.hf_inference import HFClient
from og_ego_prim.models.server_inference import ServerClient
from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.primitives import VALID_PRIMITIVES
from og_ego_prim.benchmark.tracker import EvalTracker
from og_ego_prim.utils.constants import WORK_DIR
from og_ego_prim.utils.prompts import *
from og_ego_prim.utils.types import StepwisePlan

from og_ego_prim.utils.constants import TASKS

class BadAgentPlanError(Exception):
    pass


def parse_output(output: str) -> Optional[StepwisePlan]:
    dethink = re.sub(r'.*?</think>', '', output, flags=re.S).strip()
    try:
        result = json.loads(dethink)
        return result
    except:
        pass
    pattern = r'```json(.*?)```'
    result = re.findall(pattern, dethink, re.DOTALL)
    
    if len(result) >= 1:
        result = result[0].strip()
        try:
            result = json.loads(result)
        except:
            result = None
        return result
    return None
