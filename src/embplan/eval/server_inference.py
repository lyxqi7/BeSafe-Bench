import os 
import random
import time
from typing import List
from google import genai
from google.genai import types
from openai import OpenAI 

from og_ego_prim.models.base_client import BaseClient
from og_ego_prim.models.image_utils import (
    encode_image, 
    guess_image_type_from_base64,
)

def read_image(image_path: str):
  with open(image_path, "rb") as f:
     return f.read()
  
class ServerClient(BaseClient):
    def model(self, prompt, image_file: List[str] | str = None, gen_args={"max_completion_tokens": 8096, "temperature": 0.0}): 
        if not image_file: 
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        else:
            if isinstance(image_file, str):  # support single and multi image
                image_file = [image_file]   
            if 'gemini_direct' in self.model_name.lower(): # support gemini api
                parts=[
                        types.Part.from_text(text=prompt)
                    ]
                for image in image_file:
                    image_base64 = read_image(image)
                    image_type = "image/png"
                    image_content = types.Part.from_bytes(
                        data=image_base64,
                        mime_type=image_type,
                    )
                    parts.append(image_content)
                contents = [
                    types.Content(
                        role="user",
                        parts=parts
                    )
                ]

                for _ in range(3):
                    result = ""
                    try:
                        for chunk in self.client.models.generate_content_stream(
                            model = self.model_name,
                            contents = contents,
                            config = self.generate_content_config,
                            ):
                            result += chunk.text
                        if len(result) == 0:  
                            continue
                        return result
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                        continue        
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                        ],
                    }
                ]
                for image in image_file:
                    image_base64 = encode_image(image)
                    image_type = guess_image_type_from_base64(image_base64)
                    messages[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base64}"
                            },
                        }
                    )
                for _ in range(3):
                    result = ""
                    try:
                        chat_completion = self.client.chat.completions.create(
                                messages=messages,
                                model=self.model_name,
                                **gen_args
                            )
                        result = chat_completion.choices[0].message.content
                        if not result :   # 避免none的出现
                            continue
                        return result
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                        continue
        return result
