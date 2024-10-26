import json
import sys
import os
import yaml
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from tools.lvm_pool import gpt4o, claude_3_5_sonnet, gemini_1_5_flash, llava_1_6, glm_v4

test_mode = {
    "GPT4o": gpt4o,
    "Claude3.5-sonnet": claude_3_5_sonnet,
    "Gemini1.5-flash": gemini_1_5_flash,
    "Llava1.6":llava_1_6,
    "GLM4v": glm_v4,
}

# Load questions and image URLs from JSON files
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config.get('test_lvm', 'GPT4o')

with open('./document/lvm_questions.json', 'r') as file:
    questions_data = json.load(file)

with open('./document/image_urls.json', 'r') as file:
    image_urls_data = json.load(file)

# Initialize answers dictionary
answers = {}

# Process each aspect and send questions to the LLM
for i, aspect_entry in enumerate(questions_data):
    aspect_name = aspect_entry['aspect']['aspect']
    questions = aspect_entry['questions'].split('\n')

    # Get image URL and prompt for the corresponding aspect
    image_url = image_urls_data[i]['image_url']
    prompt = image_urls_data[i]['prompt']

    answers[aspect_name] = []
    for question in questions:
        if question.strip():  # Check if the question is not an empty string
            message = f"Based on the image, prompt, and the question, provide an answer and a rationale clearly.\nPrompt: {prompt}\nQuestion: {question}"
            answer = model_name(message, image_url)
            print(f"Question: {question}\nAnswer: {answer}\n")
            answers[aspect_name].append({'question': question, 'answer': answer})

# Save answers to file
answers_file_path = './document/lvm_answers.json'
with open(answers_file_path, 'w') as file:
    json.dump(answers, file, indent=4)

print("Done!")
