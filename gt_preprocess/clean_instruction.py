import re
def clean_instruction(text):
    text = text.rstrip()
    if text.startswith(' '):
        text = text.lstrip()
    text = re.sub(r'\s+', ' ', text)
    if not text.endswith('.'):
        text += '.'

    if "The instruction is: " in text:
        print("Invalid text: ",text)
        text = text.split("The instruction is: ")[1]
        print("Valid text: ",text)

    if "The target instruction means" in text:
        print("Invalid text: ",text)
        text = text.split("The target instruction means")[1]
        print("Valid text: ",text)

    if "The instruction translates to:" in text:
        print("Invalid text: ",text)
        text = text.split("The instruction translates to: ")[1]
        print("Valid text: ",text)

    if "The instruction in English would be:" in text:
        print("Invalid text: ",text)
        text = text.split("The instruction in English would be: ")[1]
        print("Valid text: ",text)

    if "Instruction for the target:" in text:
        print("Invalid text: ",text)
        text = text.split("target: ")[1]
        print("Valid text: ",text)

    if "Instruction translation: " in text:
        print("Invalid text: ",text)
        text = text.split("Instruction translation: ")[1]
        print("Valid text: ",text)
    
    if "Translation: " in text:
        print("Invalid text: ",text)
        text = text.split("Translation: ")[1]
        print("Valid text: ",text)

    if "instruction: " in text:
        print("Invalid text: ",text)
        text = text.split("instruction: ")[1]
        print("Valid text: ",text)

    if "Instruction: " in text:
        print("Invalid text: ",text)
        text = text.split("Instruction: ")[1]
        print("Valid text: ",text)
    return text