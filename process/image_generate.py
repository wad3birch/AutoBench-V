import requests
import json
import yaml
import sys
import os

parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from config.configuration import config
from tools.diffusion_model import stable_diffusion_3, openjourney_v4, dalle_3, sdxl
from tools.tifa import generate_and_evaluate_image

# Model function mapping
model_functions = {
    "stable_diffusion_3": stable_diffusion_3,
    "openjourney_v4": openjourney_v4,
    "dalle_3": dalle_3,
    "sdxl": sdxl
    # Add other models here...
}

# Load configuration from YAML file
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config.get('diffusion_model', 'dalle_3')
user_input = config.get('user_input')
level = config.get('level')

# Load prompts from JSON file
with open(f'./document/{user_input}_understanding/{level}_{user_input}_image_prompts.json', 'r') as file:
    prompts = json.load(file)

# Directory for storing images
image_dir = os.path.join(os.curdir, 'images')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# Initialize lists for storing image URLs and results
image_urls_list = []
results_list = []

# File paths for saving data
image_urls_file_path = './document/image_urls.json'
results_file_path = f'./document/{user_input}/{level}_{user_input}_image_results_with_details.json'

def save_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def generate_image(aspect, prompt, index, model_name):
    if model_name not in model_functions:
        raise ValueError(f"Model {model_name} not supported.")
    
    model_function = model_functions[model_name]
    prompt = prompt + 'Generate images from the observer\'s point of view and orientation'
    image_urls = model_function(prompt)
    
    if not image_urls:
        raise ValueError("No image URLs returned from the model.")

    if model_name == "dalle_e_3":
        image_url = image_urls
    else:
        image_url = image_urls[0]
    
    image_urls_list.append({'aspect': aspect, 'prompt': prompt, 'image_url': image_url})
    image_path = os.path.join(image_dir, f'generated_image_{index}.png')
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)
    
    return image_url

# Generate images for each prompt and save URLs
for i, entry in enumerate(prompts):
    aspect = entry['aspect']
    prompt = entry['prompt']
    score = 0
    
    # while score < 0.8:
    image_url = generate_image(aspect, prompt, i, model_name)
    result = generate_and_evaluate_image(image_url, prompt)
    score = result['score']
    
    # Save image URL details after each generation
    save_to_file(image_urls_list, image_urls_file_path)

    results_list.append({
        'aspect': aspect,
        'prompt': prompt,
        'image_url': image_url,
        'score': score,
        'questions': result.get('responses', [])
    })

    # Save results details after each generation
    save_to_file(results_list, results_file_path)

print("Images generated and URLs saved to image_urls.json")
