from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import torch

# Load model and tokenizer
model_name = "GRMenon/mental-health-mistral-7b-instructv0.2-finetuned-V2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Function to generate chatbot responses
def chat(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Gradio UI
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="Mental Health Chatbot")
iface.launch()
