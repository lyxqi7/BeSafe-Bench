# evaluation/safety/llm_judge.py

from openai import OpenAI


class LLMJudge:

    def __init__(
        self,
        api_key="",
        api_base="",
        model_name="gpt-5",
        temperature=0.0,
        max_tokens=512,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.model = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()
