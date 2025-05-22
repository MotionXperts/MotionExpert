import re
def clean_instruction(text):
    text = text.rstrip()
    if text.startswith(' '):
        text = text.lstrip()
    text = re.sub(r'\s+', ' ', text)
    if not text.endswith('.'):
        text += '.'

    if "instruction: " in text:
        print("Invalid text: ",text)
        text = text.split("instruction: ")[1]
        print("Valid text: ",text)

    if "Instruction: " in text:
        print("Invalid text: ",text)
        text = text.split("Instruction: ")[1]
        print("Valid text: ",text)
    return text