import json
import yaml
import os
import sys
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from tools.message_sender import send_message

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json(data,file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        
config = load_config('./config/config.yaml')
guide_prompt_template = config['guidance_prompt']
user_input = config.get('user_input')
load_json_path = f'/Users/wad3/Downloads/paper/visual_autobench/code/document/{user_input}/{user_input}_aspects.json'
data = read_json('/Users/wad3/Downloads/paper/visual_autobench/code/document/reasoning_capacity/reasoning_capacity_aspects.json')

guidance=[]
for aspect in data:
    prompt = guide_prompt_template.format(aspect=aspect['aspect'], introduction=aspect['introduction'])
    response = send_message(prompt, image_url=None)
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("Aspect:"):
            current_content = {"aspect": line[len('Aspect:'):].strip()}
        if line.startswith("Introduction:"):
            current_content["introduction"] = line[len('Introduction:'):].strip()
        if line.startswith("Guidance:") and current_content:
            current_content["guidance"] = line[len("Guidance:"):].strip()
            guidance.append(current_content)
            current_content = None

write_json(guidance, f'/Users/wad3/Downloads/paper/visual_autobench/code/document/{user_input}/{user_input}_guidance.json')


    