import openai
import os
from dotenv import load_dotenv
from pathlib import Path

def BX_get_system_prompt_ENG() :
    path = f"./BX_template/ENG_prompt.txt"
    text_template = open(path).read()
    return text_template

def BX_translate_to_english(instruction):
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    openai.api_key = os.getenv("CINDY_OPENAI_KEY")
    openai.organization = os.getenv("CINDY_ORGANIZATION")

    print("Translate to english...")

    system_prompt = BX_get_system_prompt_ENG()
    print(system_prompt)

    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Target instruction: {instruction}'}
        ]
    )
    print(f'Target instruction: {instruction}')
    translate_instruction = response.choices[0].message.content
    return translate_instruction