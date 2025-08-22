from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import re
import json


load_dotenv()

_gemini_client = genai.Client()
_openai_client = OpenAI()


def getnerate_with_gemini(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1024,
    response_shcema = None,
):
    config = genai.types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        max_output_tokens=max_output_tokens,
        thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
    )
    if response_shcema:
        config.response_schema = response_shcema

    response = _gemini_client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )

    response_text = response.text
        
    # LLM 응답에 포함된 마크다운을 제거
    if response_shcema:
        if "```" in response_text:
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                response_text = match.group(0)
        response_text = json.loads(response_text)
    
    return response_text


def generate_with_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 1024,
    text_format = None,
):
    response = _openai_client.responses.parse(
        model=model,
        input=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt },
        ],
        max_output_tokens=max_output_tokens,
        text_format=text_format,
    )

    if text_format:
        return response.output_parsed.model_dump()
    
    return response.output
