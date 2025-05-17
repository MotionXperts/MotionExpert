import openai
import os
from dotenv import load_dotenv
from pathlib import Path

def BX_get_system_prompt(motion_type, index) :
    path = f"./BX_template/augmentation_prompt_{index}.txt"
    text_template = open(path).read()

    prompt      = text_template.replace('{{motion_type}}', motion_type)
    return prompt

def BX_augmentation(instruction, motion_type, index):
    env_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    openai.api_key = os.getenv("CINDY_OPENAI_KEY")
    openai.organization = os.getenv("CINDY_ORGANIZATION")

    print("Running augmentation...")

    system_prompt = BX_get_system_prompt(motion_type, index)
    print(system_prompt)

    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f'Target instruction: {instruction}'}
        ]
    )
    print(f'Target instruction: {instruction}')
    aug_instruction = response.choices[0].message.content
    return aug_instruction
