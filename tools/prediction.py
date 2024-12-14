# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)

def calculate_next_token_probability(question, possible_answers=['A', 'B', 'C', 'D']):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Your task is to answer multiple choice questions and give a direct answer (A or B or C or D)."},
        {"role": "user", "content": question + tokenizer.eos_token}
    ]

    # Find the token with the highest probability
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text + tokenizer.eos_token, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    
    probabilities_dict = {}
    for answer in possible_answers:
        answer_id = tokenizer.convert_tokens_to_ids(answer)
        probabilities_dict[answer] = probabilities[0, answer_id].item()
    
    return probabilities_dict