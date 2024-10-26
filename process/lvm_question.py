import os
import json
import yaml
import sys
import os
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from config.azure_client import config
from tools.message_sender import send_message

# Load configuration from YAML file
config_file_path = './config/config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

questions_per_aspect = config.get('questions_per_aspect')  # Default to 5 if not specified
subjective_prompt_template = config.get('subjective_questions_prompt')
objective_prompt_template = config.get('objective_questions_prompt')

with open('./document/image_urls.json', 'r') as file:
    image_data = json.load(file)

def generate_question(image_url, prompt, aspect, question_type):
    if question_type == "subjective":
        message = subjective_prompt_template.format(prompt=prompt, image_url=image_url, aspect=aspect)
    else:
        message = objective_prompt_template.format(prompt=prompt, image_url=image_url, aspect=aspect)
    
    response = send_message(message, image_url)
    return response.strip()

# Generate questions for each image
questions_list = []
for i, entry in enumerate(image_data):
    prompt = entry['prompt']
    print(f"Prompt: {prompt}")
    image_url = entry['image_url']
    aspect = entry['aspect']
    print(f"Aspect: {aspect}")
    aspect_questions = []
    
    question_type = "subjective" if i % 2 == 0 else "objective"
    question = generate_question(image_url, prompt, question_type, aspect)
    print(f"Question: {question}")
    
    questions_list.append({
        "aspect": aspect,
        "prompt": prompt,
        "questions": question
    })

# Save the questions to a JSON file
questions_file_path = './document/lvm_questions.json'
with open(questions_file_path, 'w') as file:
    json.dump(questions_list, file, indent=4)

print("Questions generated and saved to lvm_questions.json")
