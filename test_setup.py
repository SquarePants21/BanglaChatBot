# checking torch setup
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if torch.cuda.is_available():
    print("Yay cuda is there nice! But make sure that you have 24GB of VRAM OR higher")
else:
    print("Well CPUS are kinda slow you know")

# Choose the model size
model_name = "microsoft/DialoGPT-medium"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Now you can use this model to generate responses
input_ids = tokenizer.encode("Hi how are you doing?"
                             + tokenizer.eos_token, return_tensors="pt")

chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
output = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
print(output)
