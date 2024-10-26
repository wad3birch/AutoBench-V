import re
import requests
import json
import yaml
import sys
import os
import random
from tqdm import tqdm
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from config.configuration import config
from tools.diffusion_model import stable_diffusion_v2_1, openjourney_v4, redshift_diffusion, kandinsky_v2_2, dreamshaper_v8, dalle_e_3, sdxl
from tools.tifa import generate_and_evaluate_image
from tools.lvm_pool import gpt4o, llava_1_6, gemini_1_5_flash, claude_3_5_sonnet, glm_v4

# Model function mapping
model_functions = {
    "stable_diffusion_v2_1": stable_diffusion_v2_1,
    "openjourney_v4": openjourney_v4,
    "redshift_diffusion": redshift_diffusion,
    "kandinsky_v2_2": kandinsky_v2_2,
    "dreamshaper_v8": dreamshaper_v8,
    "dalle_e_3": dalle_e_3,
    "sdxl": sdxl
    # Add other models here...
}

lvm_functions = {
    "gpt4o": gpt4o,
    "llava_1_6": llava_1_6,
    "gemini_1_5_flash": gemini_1_5_flash,
    "claude_3_5_sonnet": claude_3_5_sonnet,
    "glm_v4": glm_v4
}

test_input = 'spatial'
sample_size = 10
# Load configuration from YAML file
with open('./config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

judge_lvm = 'gpt4o'
preprocess = False
model_name = config.get('diffusion_model', 'dalle_e_3')
lvm_set = config.get('lvm_set', [])
subjective_prompt_template = config.get('subjective_questions_prompt')
objective_prompt_template = config.get('objective_questions_prompt')
eval_lvm_prompt_template = config.get('eval_lvm_prompt_template')
eval_model_response_prompt_template = config.get('eval_model_response_prompt_template')
image_dir = os.path.join(os.curdir, 'document', f"{test_input}_understanding")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

with open(os.path.join(image_dir, f"{test_input}_image_prompts.json"), 'r') as file:
    prompts = json.load(file)

random.seed(42)
sample_prompts = random.sample(prompts, sample_size)

image_urls_list = []
results_list = []
def generate_image(aspect, prompt, index, model_name):
    if model_name not in model_functions:
        raise ValueError(f"Model {model_name} not supported.")
    
    model_function = model_functions[model_name]
    image_urls = model_function(prompt)
    
    if not image_urls:
        raise ValueError("No image URLs returned from the model.")

    if model_name == "dalle_e_3":
        image_url = image_urls
    else:
        image_url = image_urls[0]
    
    image_urls_list.append({'aspect': aspect, 'prompt': prompt, 'image_url': image_url})
    image_path = os.path.join(image_dir, f'{test_input}_sample_{index}.png')
    generated_image = requests.get(image_url).content
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)
    
    return image_url

def generate_question(image_url, prompt, aspect, question_type, model_function):
    if question_type == "subjective":
        message = subjective_prompt_template.format(prompt=prompt, image_url=image_url, aspect=aspect)
    else:
        message = objective_prompt_template.format(prompt=prompt, image_url=image_url, aspect=aspect)
    response = model_function(message, image_url)
    return response.strip()

def generate_answer(image_url, question, lvm_function):
    message = eval_lvm_prompt_template.format(question=question)
    response = lvm_function(message, image_url)
    print(f"Generated answer used {lvm_function.__name__}")
    return response.strip()

def extract_bracket_content(text):
    pattern = r'\[\[(.*?)\]\]'
    matches = re.findall(pattern, text)
    try:
        return matches[-1]
    except:
        return ''
    
question_list = []
answer_list = []
# generate images and get the evaluation scores, then generate the questions
def generate_qa():
    for i, prompt in tqdm(enumerate(sample_prompts), desc="Generating images and questions", total=sample_size):
        aspect = prompt['aspect']
        prompt_text = prompt['prompt']
        index = i + 1
        score = 0
        attempts = 0

        # align
        while score < 0.5 and attempts < 5:
            image_url = generate_image(aspect=aspect, prompt=prompt_text, index=index, model_name=model_name)
            result = generate_and_evaluate_image(image_url=image_url, prompt=prompt_text)
            score = result['score']
            attempts += 1

        if attempts >= 5:
            continue

        # genarate questions
        question_type = "subjective" if i % 2 == 0 else "objective"
        lvm_function = lvm_functions[judge_lvm]
        question = generate_question(image_url, prompt_text, aspect, question_type, lvm_function)
        
        # generate answers
        answers = []
        for lvm in lvm_set:
            lvm_function = lvm_functions[lvm]
            answer = generate_answer(image_url, question, lvm_function)
            answers.append({'lvm': lvm,
                            'answer': answer})
        
        answer_list.append(
            {
                'user_input': test_input,
                'aspect': aspect,
                'prompt': prompt_text,
                'image_url': image_url,
                'question': question,
                'answers': answers
            }
        )

        # evaluate answers
        eval_result = []
        for item in enumerate(answer_list):
            aspect = item['aspect']
            prompt = item['prompt']
            image_url = item['image_url']
            question = item['question']
            answers = item['answers']
            scores = []
            score_results = []
            sum = 0
            for answer in answers:
                lvm = answer['lvm']
                response = answer['answer']
                message = eval_model_response_prompt_template.format(description=prompt, question=question, model_answer=response)
                content = lvm_functions[judge_lvm](message, image_url)
                score = float(extract_bracket_content(content))
                sum += float(score)
                scores.append({'lvm': lvm,
                               'content': content,
                                'score': score})
            score_results.append(sum/len(answers))
            eval_result.append(
                {
                    'user_input': test_input,
                    'aspect': aspect,
                    'prompt': prompt,
                    'image_url': image_url,
                    'question': question,
                    'answers': answers,
                    'scores': scores
                }
            )
        print(score_results)
        with open(os.path.join(image_dir, f"{test_input}_evaluation_results.json"), 'w') as file:
            json.dump(eval_result, file, indent=4)

    print(f"Analysis completed. {len(answer_list)} questions and answers generated.")

def adjust_level():
    with open(os.path.join(image_dir, f"{test_input}_evaluation_results.json"), 'r') as file:
        eval_result = json.load(file)
    score_results = []
    for item in eval_result:
        avg = []
        for score in item['scores']:
            avg.append(score['score'])
        score_results.append(sum(avg)/len(avg))
    print(score_results)


    
controller_function = {
    True: generate_qa,
    False: adjust_level
}

if __name__ == '__main__':
    controller_function[preprocess]()
