from evaluation.safety.llm_judge import LLMJudge
from evaluation.safety.run_safety_eval import run_safety_eval

llm_judge = LLMJudge(
    api_key="",
    api_base="",
    model_name="gpt-5",
)

summary = run_safety_eval(
    agent_log_dir="logs/evaluation/gpt4",
    yaml_root="yaml_output",
    llm_judge=llm_judge,
)

print(summary)