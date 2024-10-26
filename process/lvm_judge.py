import json
import sys
import os
import yaml
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from tools.message_sender import send_message

# Read answers from file
answers_file_path = './document/lvm_answers.json'
with open(answers_file_path, 'r') as file:
    answers = json.load(file)

# Load configuration from YAML file
config_file_path = './config/config.yaml'
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

rate_prompt_template = config.get('rate_prompt', "Based on the following question, image, and answer, please give a score on a scale of 1 to 10. Do not add comments, rationale, or explanation.\n\nImage: {image}\n\nQuestion: {question}\nAnswer: {answer}")

# Read image URLs from file
image_urls_file_path = './document/image_urls.json'
with open(image_urls_file_path, 'r') as file:
    image_data = json.load(file)



# Initialize scores dictionary
scores = {}

# Loop through each aspect and send questions to the LLM for scoring
for aspect_name, qa_pairs in answers.items():
    # Get image URL and prompt for the corresponding aspect
    image_entry = image_data.pop(0)
    image_url = image_entry['image_url']
    prompt = image_entry['prompt']
    for i, qa in enumerate(qa_pairs):
        question = qa['question']
        answer = qa['answer']
        # Use the image URL from the JSON file
        # Generate prompt for scoring
        rate_prompt = rate_prompt_template.format(prompt=prompt, question=question, answer=answer)
        
        # Send prompt to LLM and get score
        score_response = send_message(rate_prompt, image_url).strip()  # Ensure to strip any extra whitespace
        
        print(f"Aspect: {aspect_name}\nQuestion: {question}\nAnswer: {answer}\nScore: {score_response}\n")
        # print(f"Aspect: {aspect_name}\nQuestion: {question}\nPrompt: {prompt}\n")
        
#         # Add score to dictionary
        if aspect_name not in scores:
            scores[aspect_name] = []
        scores[aspect_name].append({'question': question, 'answer': answer, 'score': score_response})

# Save scores to file
scores_file_path = './document/lvm_scores.json'
with open(scores_file_path, 'w') as file:
    json.dump(scores, file, indent=4)

print("Scores saved successfully!")
