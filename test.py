'''
from transformers import AutoTokenizer, AutoModel, utils
from Viz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = "microsoft/xtremedistil-l12-h384-uncased"  # Find popular HuggingFace models here: https://huggingface.co/models
input_text = "The cat sat on the mat"  
model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view_data = model_view(attention, tokens,  html_action='return')  # Display model view

with open("/home/weihsin/projects/MotionExpert/HTML/model_view.html", 'w') as file:
    file.write(model_view_data.data)
    print("HI")
'''
from transformers import AutoTokenizer, AutoModel, utils
from Viz import head_view

utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)

inputs = tokenizer.encode("The cat sat on the mat", return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

html_head_view = head_view(attention, tokens, html_action='return')
print(tokens)
with open("/home/weihsin/projects/MotionExpert/HTML/head_view.html", 'w') as file:
    file.write(html_head_view.data)