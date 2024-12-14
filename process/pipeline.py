import json
import yaml
import sys
import os
import aiofiles
import asyncio
import uuid
import re
import csv
import aiohttp
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm_asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.tifa import generate_and_evaluate_image
from tools.diffusion_model import async_dalle_3, async_openjourney_v4, async_sdxl, async_stable_diffusion, async_stable_diffusion_3, async_flux_pro, async_flux_1_1_pro
from tools.lvm_pool import async_gpt4o, async_gemini_1_5_flash, async_claude_3_5_sonnet, async_claude_3_haiku, async_glm_4v, async_gpt4o_mini
from tools.prediction import calculate_next_token_probability

async_diffusion_function = {
    "dalle_e_3": async_dalle_3,
    "openjourney_v4": async_openjourney_v4,
    "sdxl": async_sdxl,
    "stable_diffusion": async_stable_diffusion,
    "stable_diffusion_3": async_stable_diffusion_3,
    "flux_pro": async_flux_pro,
    "flux_1_1_pro": async_flux_1_1_pro
}

async_lvm_function = {
    "gpt-4o": async_gpt4o,
    "gpt4o_mini": async_gpt4o_mini,
    "gemini_1_5_flash": async_gemini_1_5_flash,
    "claude_3_5_sonnet": async_claude_3_5_sonnet,
    "claude_3_haiku": async_claude_3_haiku,
    "glm_4v": async_glm_4v,
    # "qwen2_vl": async_qwen_2_vl
}

lvm_func_to_name = {
    "async_gpt4o": "GPT-4o",
    "async_gpt4o_mini": "GPT-4o-Mini",
    "async_gemini_1_5_flash": "Gemini-1.5-Flash",
    "async_claude_3_5_sonnet": "Claude-3.5-Sonnet",
    "async_claude_3_haiku": "Claude-3-Haiku",
    "async_glm_4v": "GLM-4v",
    "asyn_llama_3_2":"LLAMA-3.2-90B",
    # "async_qwen_2_vl":"Qwen2_VL"
}

async def load_config(config_file_path):
    async with aiofiles.open(config_file_path, 'r', encoding='utf-8') as file:
        content = await file.read()
    return yaml.safe_load(content)

async def load_json(file_path):
    async with aiofiles.open(file_path, 'r') as file:
        content = await file.read()
    return json.loads(content)

async def save_json(data, file_path):
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(json.dumps(data, indent=4))

def save_plot(fig, path, dpi=300):
    fig.savefig(path, format='png', dpi=dpi)

async def download_image(session, url, save_path):
    try:
        async with session.get(url, ssl=False) as response:
            response.raise_for_status()
            content = await response.read()
            async with aiofiles.open(save_path, 'wb') as file:
                await file.write(content)
    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        raise

async def save_aspects(aspects, aspects_file_path):
    os.makedirs(os.path.dirname(aspects_file_path), exist_ok=True)
    async with aiofiles.open(aspects_file_path, 'w') as file:
        await file.write(json.dumps(aspects, indent=4))

async def generate_fine_grained_aspects(user_input, aspect_count, aspect_prompt):
    message = aspect_prompt.format(aspect_count=aspect_count)
    aspects_response = await async_lvm_function['gpt-4o'](message)
    print(aspects_response)

    aspects = []
    current_aspect = None
    for line in aspects_response.split('\n'):
        line = line.strip()  # Remove leading and trailing whitespace
        if line.startswith("Fined-grained Aspect:"):
            current_aspect = {"aspect": line[len("Fined-grained Aspect:"):].strip()}
        elif line.startswith("Introduction:") and current_aspect:
            current_aspect["introduction"] = line[len("Introduction:"):].strip()
            aspects.append(current_aspect)  # Save the current aspect once complete
            current_aspect = None  # Reset for the next aspect

    aspects_file_path = f'./document/{user_input}/{user_input}_aspects.json'
    await save_aspects(aspects, aspects_file_path)

    print("Fine-grained aspects generated and saved successfully!")

async def generate_guidance(data, guide_prompt_template):
    guidance = []
    for aspect in data:
        prompt = guide_prompt_template.format(aspect=aspect['aspect'], introduction=aspect['introduction'])
        response = await async_lvm_function['gpt-4o'](prompt)
        current_content = None
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
    return guidance

async def generate_prompt_with_topic_words(aspects, image_prompt_template, level, num_prompts_per_aspect):
    prompts = []
    all_topic_word_degrees = []  # List to store topic words and their degrees
    
    for aspect, introduction, guidance in aspects:
        G = nx.Graph()
        used_words = set()
        degrees_over_4 = []
        degrees_over_5 = []
        degrees_over_6 = []

        for round_num in range(num_prompts_per_aspect):
            used_words_str = ', '.join(used_words)
            image_description = image_prompt_template.format(aspect=aspect, introduction=introduction, level=level, used_words_str=used_words_str, guidance=guidance)
            prompt_response = await async_lvm_function['gpt-4o'](image_description)
            print(prompt_response)

            prompt = None
            topic_word = None
            key_words = None
            retry_times = 3
            try:
                prompt_lines = prompt_response.split('\n')
            except:
                for i in range(retry_times):
                    prompt_response = await async_lvm_function['gpt-4o'](image_description)
                    prompt_lines = prompt_response.split('\n')
                    if prompt_lines:
                        break

            for line in prompt_lines:
                if line.startswith("Prompt:"):
                    prompt = line[len("Prompt:"):].strip()
                if line.startswith("Topic word:"):
                    topic_word = line[len("Topic word:"):].strip().lower()
                if line.startswith("Key word:") or line.startswith("Key words:"):
                    key_words = line[len("Key words:"):].strip().lower()
                    key_words_list = [word.strip() for word in key_words.split(',')]
                    break
            
            if prompt and topic_word and key_words:
                prompts.append({
                    "aspect": aspect,
                    "prompt": prompt,
                    "topic_word": topic_word,
                    "key_words": key_words
                })
                G.add_node(topic_word)
                for key_word in key_words_list:
                    G.add_node(key_word)
                    G.add_edge(topic_word, key_word)

                degree_dict = dict(G.degree())
                degrees_over_4.append(sum(deg > 4 for deg in degree_dict.values()))
                degrees_over_5.append(sum(deg > 5 for deg in degree_dict.values()))
                degrees_over_6.append(sum(deg > 6 for deg in degree_dict.values()))

                top_nodes = [node for node, degree in sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)[:round_num + 1]]
                used_words.update(top_nodes)

                all_topic_word_degrees.append((topic_word, degree_dict[topic_word]))

                print(f"Round {round_num + 1} - Top node(s) selected: {top_nodes}")

        print(f"Final degrees for aspect '{aspect}': {dict(G.degree())}")

    return prompts, all_topic_word_degrees

async def generate_prompts(user_input):
    config_file_path = './config/config.yaml'
    config = await load_config(config_file_path)

    guide_prompt_template = config['guidance_prompt']
    aspect_count = 24
    aspect_prompt = config.get(f'{user_input}_prompt')

    # Generate fine-grained aspects
    await generate_fine_grained_aspects(user_input, aspect_count, aspect_prompt)

    # Load aspects from the generated JSON file
    aspects_file_path = f'./document/{user_input}/{user_input}_aspects.json'
    aspects_data = await load_json(aspects_file_path)

    # Generate guidance content
    guidance = await generate_guidance(aspects_data, guide_prompt_template)

    # Save guidance content to a JSON file
    guidance_file_path = f'./document/{user_input}/{user_input}_guidance.json'
    await save_json(guidance, guidance_file_path)
    print(f"{user_input} guidance generated and saved successfully!")

    for level in ['easy', 'medium', 'hard']:
        image_prompt_template = config.get('difficulty_control_image_prompt')

        aspects_file_path = f'./document/{user_input}/{user_input}_guidance.json'
        aspects_data = await load_json(aspects_file_path)

        aspects = [(aspect_data['aspect'], aspect_data['introduction'], aspect_data['guidance']) for aspect_data in aspects_data]

        generated_prompts, topic_word_degrees = await generate_prompt_with_topic_words(aspects, image_prompt_template, level, 10)

        prompts_file_path = f'./document/{user_input}/prompts/{level}_basic_image_prompts.json'
        if not os.path.exists(f'./document/{user_input}/prompts'):
            os.makedirs(f'./document/{user_input}/prompts')
        await save_json(generated_prompts, prompts_file_path)

        csv_file_path = f'./document/{user_input}/{level}_topic_word_degrees.csv'

        print(f"{user_input} {level} prompts generated and saved successfully!")

async def generate_single_image(item, level, image_prompt_folder, retry_attempts=3):
    prompt = item['prompt']
    aspect = item['aspect']
    id = str(uuid.uuid4())
    for attempt in range(retry_attempts):
        try:
            prompt = "please generate a picture from the perspective of an observer" + prompt
            image_url = await async_diffusion_function['flux_1_1_pro'](prompt)
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                image_folder = f'{image_prompt_folder}/extracted_images/{level}'
                os.makedirs(image_folder, exist_ok=True)
                image_path = f'{image_folder}/{id}.png'
                await download_image(session, image_url, image_path)
                return {
                    "id": id,
                    "aspect": aspect,
                    "prompt": prompt,
                    "image_url": image_url[0],
                    "image_path": os.path.abspath(image_path),
                    'level': level,
                    'model': 'flux_1_1_pro'
                }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_attempts - 1:
                return None

async def generate_images(user_input):
    image_prompt_folder = f'./document/{user_input}/prompts/'
    store_folder = f'./document/{user_input}/'
    os.makedirs(image_prompt_folder, exist_ok=True)
    os.makedirs(store_folder, exist_ok=True)
    
    for level in ['easy', 'medium', 'hard']:
        image_prompt_file = f'{image_prompt_folder}/{level}_basic_image_prompts.json'
        image_prompt_data = await load_json(image_prompt_file)
        
        tasks = []
        for item in image_prompt_data:
            task = asyncio.create_task(generate_single_image(item, level, store_folder))
            tasks.append(task)
        
        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        save_file = f'{store_folder}/image_json/{level}_images.json'
        if not os.path.exists(f'{store_folder}/image_json'):
            os.makedirs(f'{store_folder}/image_json')
        await save_json(save_data, save_file)
        print(f'{level} photos generated and saved successfully!')

async def align_single_image(item, level, retry_attempts=3, threshold=0.0, align_attempts=3):
    aspect = item['aspect']
    image_path = item['image_path']
    prompt = item['prompt']
    if level == 'easy':
        threshold = 0.0
    for attempt in range(retry_attempts):
        try:
            score, results = await generate_and_evaluate_image(image_path, prompt)
            align_attempt = 0
            while score < threshold and align_attempt < align_attempts:
                score, results = await generate_and_evaluate_image(image_path, prompt)
                align_attempt += 1
            if score >= threshold:
                return {
                    "aspect": aspect,
                    "prompt": prompt,
                    "image_path": image_path,
                    'level': level,
                    'model': 'gpt4o',
                    'score': score,
                    'align_results': results
                }
            else:
                return None
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}.")
            if attempt == retry_attempts - 1:
                return None

async def align_images(user_input):
    image_prompt_folder = f'./document/{user_input}'
    os.makedirs(image_prompt_folder, exist_ok=True)
    for level in ['easy', 'medium', 'hard']:
        image_prompt_file = f'{image_prompt_folder}/image_json/{level}_images.json'
        image_prompt_data = await load_json(image_prompt_file)
        
        tasks = []
        semaphore = asyncio.Semaphore(20)

        async def sem_task(item, level):
            async with semaphore:
                return await align_single_image(item, level)

        for item in image_prompt_data:
            task = asyncio.create_task(sem_task(item, level))
            tasks.append(task)
        
        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]

        save_file = f'{image_prompt_folder}/aligned_image_json/{level}_aligned_images.json'
        if not os.path.exists(f'{image_prompt_folder}/aligned_image_json'):
            os.makedirs(f'{image_prompt_folder}/aligned_image_json')
        await save_json(save_data, save_file)
        print(f'{level} images aligned and saved successfully! Totally {len(save_data)} images aligned. Align rate: {len(save_data) / len(image_prompt_data)}')

async def gen_single_question(level, item, objective_question_prompt, retry_attempts=3):
    aspect = item['aspect']
    image_path = item['image_path']
    prompt = item['prompt']
    need_elements = False
    if item['score'] == 1:
        elements = "None"
    else:
        need_elements = True
        for result in item['align_results']['responses']:
            if result['llm_answer'] != result['correct_answer']:
                elements = f"{result['element_type']}: {result['element']}"
                break
    for attempt in range(retry_attempts):
        try:
            objective_prompt = objective_question_prompt.format(aspect=aspect, elements=elements, level=level, prompt=prompt)
            objective_response = await async_gpt4o(objective_prompt)
            objective_reference_answer = json.loads(objective_response)['reference_answer']
            objective_question = json.loads(objective_response)['question'] + '\n' + json.dumps(json.loads(objective_response)['options'])
            return {
                "aspect": aspect,
                "prompt": prompt,
                "image_path": image_path,
                'level': level,
                'model': 'gpt4o',
                'objective_question': objective_question,
                'objective_reference_answer': objective_reference_answer,
                'need_elements': need_elements,
            }
        except Exception as e:
            print(f"Attemp {attempt + 1} failed: {e}")
            if attempt == retry_attempts - 1:
                return None
    
    return None

async def generate_questions(user_input):
    image_prompt_folder = f'./document/{user_input}/'
    os.makedirs(image_prompt_folder, exist_ok=True)
    config_file_path = './config/config.yaml'
    config = await load_config(config_file_path)
    for level in ['easy','medium','hard']:
        image_prompt_file = f'{image_prompt_folder}/aligned_image_json/{level}_aligned_images.json'
        image_prompt_data = await load_json(image_prompt_file)
        
        tasks = []
        semaphore = asyncio.Semaphore(20)

        async def sem_task(item, level, objective_question_prompt):
            async with semaphore:
                return await gen_single_question(level, item, objective_question_prompt)

        for item in image_prompt_data:
            objective_question_prompt = config.get('objective_question_prompt')
            task = asyncio.create_task(sem_task(item, level, objective_question_prompt))
            tasks.append(task)
        
        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        save_file = f'{image_prompt_folder}/questions/{level}_questions.json'
        if not os.path.exists(f'{image_prompt_folder}/questions'):
            os.makedirs(f'{image_prompt_folder}/questions')
        await save_json(save_data, save_file)
        print(f'{level} questions generated and saved successfully! Totally {len(save_data)} questions generated. Generate rate: {len(save_data) / len(image_prompt_data)}')

async def adjust_questions(user_input, weights=[0.25, 0.25, 0.25, 0.25]):
    questions_folder = f'./document/{user_input}/questions'
    os.makedirs(questions_folder, exist_ok=True)
    for level in ['easy', 'medium', 'hard']:
        questions_file = f'{questions_folder}/{level}_questions.json'
        questions_data = await load_json(questions_file)
        options = ['A', 'B', 'C', 'D']
        random.seed(42)
        answer_sequence = random.choices(options, weights, k=len(questions_data))
        for i, item in enumerate(questions_data):
            question = item["objective_question"]
            correct_answer = item["objective_reference_answer"]
            new_answer = answer_sequence[i]
            question_text, options_text = question.split("\n", 1)
            options_dict = json.loads(options_text)
            correct_answer_text = options_dict[correct_answer]
            options_dict[correct_answer], options_dict[new_answer] = options_dict[new_answer], correct_answer_text
            new_question = f"{question_text}\n" + json.dumps(options_dict, ensure_ascii=False)
            item["objective_question"] = new_question
            item["objective_reference_answer"] = new_answer
        save_file = f'{questions_folder}/{level}_questions.json'
        await save_json(questions_data, save_file)
        print(f'{level} questions adjusted and saved successfully! Weighted answers: {weights}')

def extract_score(text):
    pattern_brackets = r'Rating:\s*\[\[(\d+(\.\d+)?)\]\]'
    pattern_direct = r'Rating:\s*(\d+(\.\d+)?)'
    
    matches_brackets = re.findall(pattern_brackets, text)
    matches_direct = re.findall(pattern_direct, text)
    
    if matches_brackets:
        try:
            return float(matches_brackets[-1][0])
        except:
            return 0.0
    
    if matches_direct:
        try:
            return float(matches_direct[-1][0])
        except:
            return 0.0
    
    return 0.0
    
def extract_choice(text):
    try:
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, text)
        return matches[0]
    except:
        return None

async def generate_single_answer(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item, retry_attempts=8):
    aspect = item['aspect']
    image_path = item['image_path']
    prompt = item['prompt']
    # subject_question = item['subjective_question']
    objective_question = item['objective_question']
    # subject_reference_answer = item['subjective_reference_answer']
    objective_reference_answer = item['objective_reference_answer']
    for attempt in range(retry_attempts):
        try:
            # subject_answer_prompt = subjective_answer_prompt.format(aspect=aspect, question=subject_question)
            objective_answer_prompt = objective_answer_prompt.format(aspect=aspect, question=objective_question)
            # subject_answer = await model_function(subject_answer_prompt, image_path)
            objective_answer = await model_function(objective_answer_prompt, image_path)
            # subjective_eval_prompt = eval_prompt.format(prompt=prompt, reference_answer=subject_reference_answer, question=subject_question, answer=subject_answer)
            # subjective_eval_answer = await async_gpt4o(subjective_eval_prompt, image_path)
            objective_choice = extract_choice(objective_answer)
            # subjective_score = extract_score(subjective_eval_answer)
            objective_score = 1 if objective_choice == objective_reference_answer else 0
            return {
                "aspect": aspect,
                "prompt": prompt,
                "image_path": image_path,
                'level': level,
                'model': model_function.__name__,
                # 'subjective_question': subject_question,
                'objective_question': objective_question,
                # 'subjective_answer': subject_answer,
                'objective_answer': objective_answer,
                'need_elements': item['need_elements'],
                # 'subjective_eval_answer': subjective_eval_answer,
                'objective_choice': objective_choice,
                # 'subjective_score': subjective_score,
                'objective_score': objective_score,
                'objective_reference_answer': objective_reference_answer
            }
        except Exception as e:
            print(f"Model: {model_function.__name__} Attempt {attempt + 1} failed: {e}")
            if attempt == retry_attempts - 1:
                return None

async def generate_answers(user_input):
    image_prompt_folder = f'./document/{user_input}/'
    os.makedirs(image_prompt_folder, exist_ok=True)
    config_file_path = './config/config.yaml'
    config = await load_config(config_file_path)
    for level in ['easy', 'medium', 'hard']:
        image_prompt_file = f'{image_prompt_folder}/questions/{level}_questions.json'
        image_prompt_data = await load_json(image_prompt_file)
        
        tasks = []
        semaphore = asyncio.Semaphore(10)
        model_scores = {model_function.__name__: {'objective': [], 'subjective': []} for model_function in async_lvm_function.values()}
        # model_scores = {model_function.__name__: {'subjective': []} for model_function in async_lvm_function.values()}
        async def sem_task(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item):
            async with semaphore:
                return await generate_single_answer(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item)
        for item in image_prompt_data:
            subjective_answer_prompt = config.get('subjective_answer_prompt')
            objective_answer_prompt = config.get('objective_answer_prompt')
            eval_prompt = config.get('eval_model_response_prompt_template')
            for model_name in async_lvm_function.keys():
                model_function = async_lvm_function[model_name]
                task = asyncio.create_task(sem_task(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item))
                tasks.append(task)
        
        results = await tqdm_asyncio.gather(*tasks)
        results = [result for result in results if result is not None]
        
        for result in results:
            model_scores[result['model']]['objective'].append(result['objective_score'])
            # model_scores[result['model']]['subjective'].append(result['subjective_score'])
        
        save_file = f'{image_prompt_folder}/answers/{level}_answers.json'
        if not os.path.exists(f'{image_prompt_folder}/answers'):
            os.makedirs(f'{image_prompt_folder}/answers')
        await save_json(results, save_file)
        print(f'{level} answers generated and saved successfully!')

        scores_file = f'{image_prompt_folder}/scores/{level}_scores.json'
        if not os.path.exists(f'{image_prompt_folder}/scores'):
            os.makedirs(f'{image_prompt_folder}/scores')
        
        avg_scores = {}
        for model_name, scores in model_scores.items():
            avg_scores[model_name] = {
                'average_objective_score': sum(scores['objective']) / len(scores['objective']) if scores['objective'] else 0,
                # 'average_subjective_score': sum(scores['subjective']) / len(scores['subjective']) if scores['subjective'] else 0,
                'objective_num': len(scores['objective']),
                # 'subjective_num': len(scores['subjective'])
            }
            print(f'Average objective score for model {model_name} at level {level}: {avg_scores[model_name]["average_objective_score"]:.2f}')
            # print(f'Average subjective score for model {model_name} at level {level}: {avg_scores[model_name]["average_subjective_score"]:.2f}')
        
        await save_json(avg_scores, scores_file)
        print(f'{level} scores generated and saved successfully!')

async def visualization_scores(user_input, ablation=False):
    parent_path = os.path.join("document", user_input, "scores")
    if ablation:
        parent_path = os.path.join("document", user_input, "ablation_study")
    difficulties = ['easy', 'medium', 'hard']
    files = [os.path.join(parent_path, f"{difficulty}_scores.json") for difficulty in difficulties]
    models = [model_name.__name__ for model_name in async_lvm_function.values()]
    model_names = [lvm_func_to_name[model_name] for model_name in models]
    final_scores = {model: {'subjective': [], 'objective': []} for model in models}
    
    for file in files:
        data = await load_json(file)
        for model in models:
            # final_scores[model]['subjective'].append(data[model]["average_subjective_score"])
            final_scores[model]['objective'].append(data[model]["average_objective_score"])

    async def plot_scores(score_type, title, ylabel, filename):
        bar_width = 0.2
        index = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, difficulty in enumerate(difficulties):
            scores = [final_scores[model][score_type][i] for model in models]
            ax.bar(index + i * bar_width, scores, bar_width, label=difficulty)
        
        ax.set_xlabel('Models')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(model_names)
        ax.legend()
        
        plt.tight_layout()
        
        visualization_path = os.path.join(f'./document/{user_input}', "visualization", filename)
        save_plot(fig, visualization_path, 300)

    os.makedirs(f"./document/{user_input}/visualization", exist_ok=True)
    if ablation:
        # await plot_scores('subjective', f'Final Subjective Scores for Ablation Study', 'Average Subjective Score', f'{user_input}_ablation_subjective_scores.png')
        await plot_scores('objective', f'Final Objective Scores for Ablation Study', 'Average Objective Score', f'{user_input}_ablation_objective_scores.png')
    else:
        # await plot_scores('subjective', f'Final Subjective Scores with user input: {user_input}', 'Average Subjective Score', f'{user_input}_subjective_scores.png')
        await plot_scores('objective', f'Final Objective Scores with user input: {user_input}', 'Average Objective Score', f'{user_input}_objective_scores.png')

    print(f"Visualization of scores for user input {user_input} saved successfully!")

async def to_csv(user_input, ablation=False):
    parent_path = os.path.join("document", user_input, "scores")
    if ablation:
        parent_path = os.path.join("document", user_input, "ablation_study")
    difficulties = ['easy', 'medium', 'hard']
    files = [os.path.join(parent_path, f"{difficulty}_scores.json") for difficulty in difficulties]

    output_path = os.path.join(f'./document/{user_input}', "all_scores.csv")
    if ablation:
        output_path = os.path.join(f'./document/{user_input}', "ablation_study_scores.csv")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['User Input', user_input]) if not ablation else writer.writerow(['User Input', user_input, 'Ablation Study'])
        writer.writerow(['Difficulty', 'Model', 'Objective Score', 'Alignment Rate'])

        for difficulty, file in zip(difficulties, files):
            data = await load_json(file)
            for model, scores in data.items():
                writer.writerow([difficulty, lvm_func_to_name[model],scores["average_objective_score"], scores["objective_num"]/240])
    print(f"{user_input} scores saved to csv located at {output_path}")

async def single_ablation_study(level, item, objective_question_prompt, subjective_question_prompt, retry_attempts=3):
    aspect = item['aspect']
    image_path = item['image_path']
    prompt = item['prompt']
    need_elements = False
    elements = "None"
    
    for attempt in range(retry_attempts):
        try:
            subjective_prompt = subjective_question_prompt.format(aspect=aspect, elements=elements, level=level, prompt=prompt)
            objective_prompt = objective_question_prompt.format(aspect=aspect, elements=elements, level=level, prompt=prompt)
            subjective_response = await async_gpt4o(subjective_prompt)
            objective_response = await async_gpt4o(objective_prompt)
            subjective_reference_answer = json.loads(subjective_response)['reference_answer']
            objective_reference_answer = json.loads(objective_response)['reference_answer']
            subjective_question = json.loads(subjective_response)['question']
            objective_question = json.loads(objective_response)['question'] + '\n' + json.dumps(json.loads(objective_response)['options'])
            return {
                "aspect": aspect,
                "prompt": prompt,
                "image_path": image_path,
                'level': level,
                'model': 'gpt4o',
                'subjective_question': subjective_question,
                'subjective_reference_answer': subjective_reference_answer,
                'objective_question': objective_question,
                'objective_reference_answer': objective_reference_answer,
                'need_elements': need_elements,
            }
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_attempts - 1:
                return None

async def ablation_study(user_input):
    image_prompt_folder = f'./document/{user_input}/'
    os.makedirs(image_prompt_folder, exist_ok=True)
    config_file_path = './config/config.yaml'
    config = await load_config(config_file_path)
    
    for level in ['easy', 'medium', 'hard']:
        image_prompt_file = f'{image_prompt_folder}/image_json/{level}_images.json'
        image_prompt_data = await load_json(image_prompt_file)
        
        tasks = []
        for item in image_prompt_data:
            objective_question_prompt = config.get('objective_question_prompt')
            subjective_question_prompt = config.get('subjective_question_prompt')
            task = asyncio.create_task(single_ablation_study(level, item, objective_question_prompt, subjective_question_prompt))
            tasks.append(task)
        
        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        save_file = f'{image_prompt_folder}/ablation_study/{level}_questions.json'
        if not os.path.exists(f'{image_prompt_folder}/ablation_study'):
            os.makedirs(f'{image_prompt_folder}/ablation_study')
        await save_json(save_data, save_file)
        print(f'{level} questions for ablation study generated and saved successfully!')

        # Generate answers for ablation study
        tasks = []
        semaphore = asyncio.Semaphore(20)
        model_scores = {model_function.__name__: {'objective': [], 'subjective': []} for model_function in async_lvm_function.values()}
        
        async def sem_task(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item):
            async with semaphore:
                return await generate_single_answer(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item)
        
        for item in save_data:
            subjective_answer_prompt = config.get('subjective_answer_prompt')
            objective_answer_prompt = config.get('objective_answer_prompt')
            eval_prompt = config.get('eval_model_response_prompt_template')
            for model_name in async_lvm_function.keys():
                model_function = async_lvm_function[model_name]
                task = asyncio.create_task(sem_task(model_function, subjective_answer_prompt, objective_answer_prompt, eval_prompt, level, item))
                tasks.append(task)
        
        results = await tqdm_asyncio.gather(*tasks)
        results = [result for result in results if result is not None]
        
        for result in results:
            model_scores[result['model']]['objective'].append(result['objective_score'])
            model_scores[result['model']]['subjective'].append(result['subjective_score'])
        
        save_file = f'{image_prompt_folder}/ablation_study/{level}_answers.json'
        if not os.path.exists(f'{image_prompt_folder}/ablation_study'):
            os.makedirs(f'{image_prompt_folder}/ablation_study')
        await save_json(results, save_file)
        print(f'{level} answers for ablation study generated and saved successfully!')

        scores_file = f'{image_prompt_folder}/ablation_study/{level}_scores.json'
        if not os.path.exists(f'{image_prompt_folder}/ablation_study'):
            os.makedirs(f'{image_prompt_folder}/ablation_study')
        
        avg_scores = {}
        for model_name, scores in model_scores.items():
            avg_scores[model_name] = {
                'average_objective_score': sum(scores['objective']) / len(scores['objective']) if scores['objective'] else 0,
                'average_subjective_score': sum(scores['subjective']) / len(scores['subjective']) if scores['subjective'] else 0,
                'objective_num': len(scores['objective']),
                'subjective_num': len(scores['subjective'])
            }
            print(f'Average objective score for model {model_name} at level {level}: {avg_scores[model_name]["average_objective_score"]:.2f}')
            print(f'Average subjective score for model {model_name} at level {level}: {avg_scores[model_name]["average_subjective_score"]:.2f}')
        
        await save_json(avg_scores, scores_file)
        print(f'{level} scores for ablation study generated and saved successfully!')

async def prediction_single(item):
    questions = item['pertubation_Q&A']
    for question in questions:
        question_text = question['question']
        answer = question['answer']
        probability_dict = calculate_next_token_probability(question_text,possible_answers=['A','B','C','D'])
        question['answer_probability'] = probability_dict
        min_option = min(probability_dict, key=probability_dict.get)
        # calculate the sum of the probability of the three wrong options
        sum_probability = sum([probability_dict[option] for option in probability_dict if option != answer])
        # if the probability of the correct answer is less than the sum of the probability of the three wrong options, then the question needs pertubation
        if min_option != question['answer'] and probability_dict[answer] > sum_probability:
            question['need_pertubation'] = True
            question['pertubation_answer'] = min_option
        else: question['need_pertubation'] = False
    return item

async def prediction(user_input):
    # for level in ['easy', 'medium', 'hard']:
    for level in ['easy']:
        # with open(f'./document/{user_input}/questions/{level}_pertubation_questions.json', 'r') as file:
        with open('document/pertubation_questions.json', 'r') as file:
            data = json.load(file)
        tasks = []
        for item in data[:10]:
            task = asyncio.create_task(prediction_single(item))
            tasks.append(task)

        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        # save_file = f'./document/{user_input}/questions/{level}_pertubation_questions.json'
        save_file = f'./document/pertubation_prediction.json'
        await save_json(save_data, save_file)
        print(f'{level} questions for pertubation generated and saved successfully!')   

async def rewrite_single(item):
    config = await load_config('config/config.yaml')
    rewrite_template = config.get(f'rewrite_prompt')
    # question = item['objective_question']
    # reference_answer = item['objective_reference_answer']
    original_prompt = item['prompt']
    # target = item['pertubation_answer']
    content = []
    item['pertubation_Q&A'] = [item for item in item['pertubation_Q&A'] if item['need_pertubation']]
    if item['pertubation_Q&A'] == []:
        return None
    for question in item['pertubation_Q&A']:
        #如果need pertubation三个都是false就不用pertubation
        content.append(["Question: " + question['question'], "Answer: " + question['answer'], "Target Answer: " + question['pertubation_answer']])
    # rewrite_prompt = rewrite_template.format(original_prompt=original_prompt, question=question, answer=reference_answer, target_answer=target)
    rewrite_prompt = rewrite_template.format(original_prompt=original_prompt, content=content)
    rewrite_response = await async_gpt4o(rewrite_prompt)
    rewrite_response = re.sub(r"```json(.*?)```", r"\1", rewrite_response, flags=re.DOTALL).strip()
    item['modified_prompt'] = json.loads(rewrite_response)['modified_prompt']
    return item

async def rewrite(user_input):
    for level in ['easy']:
        # with open(f'./document/{user_input}/questions/{level}_pertubation_questions.json', 'r') as file:
        with open('document/pertubation_prediction.json', 'r') as file:
            data = json.load(file)
        tasks = []
        for item in data:
            task = asyncio.create_task(rewrite_single(item))
            tasks.append(task)

        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        # save_file = f'./document/{user_input}/questions/{level}_modified_questions.json'
        save_file = f'./document/pertubation_modified.json'
        await save_json(save_data, save_file)
        print(f'{level} questions for pertubation generated and saved successfully!')   
    print('easy questions rewrited successfully!')

async def pertubation_single_question(item):
    prompt = item['prompt']
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    pertubation_question_template = config.get('pertubate_question_prompt')
    pertubation_question_prompt = pertubation_question_template.format(description=prompt)
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            pertubation_question_response = await async_gpt4o(pertubation_question_prompt)
            pertubation_question_response = re.sub(r"```json(.*?)```", r"\1", pertubation_question_response, flags=re.DOTALL).strip()
            print(pertubation_question_response)
            pertubation_question_response = json.loads(pertubation_question_response)
            item['pertubation_Q&A'] = pertubation_question_response
            return item
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retry_attempts - 1:
                return None

async def pertubation_question(user_input):
    for level in ['easy', 'medium', 'hard']:
        with open(f'./document/{user_input}/prompts/{level}_{user_input}_image_prompts.json', 'r') as file:
            data = json.load(file)
        tasks = []
        for item in data:
            task = asyncio.create_task(pertubation_single_question(item))
            tasks.append(task)

        save_data = await tqdm_asyncio.gather(*tasks)
        save_data = [data for data in save_data if data is not None]
        save_file = f'./document/{user_input}/questions/{level}_pertubation_questions.json'
        await save_json(save_data, save_file)
        print(f'{level} questions for pertubation generated and saved successfully!')   
    print('easy questions rewrited successfully!')

if __name__ == '__main__':
    user_input = 'basic_understanding'
    generate_type = "alignment"
    excute_function = {
        "aspect":generate_fine_grained_aspects,
        "guideline":generate_guidance,
        "prompts": generate_prompts,
        "images": generate_images,
        "alignment": align_images,
        "questions": generate_questions,
        "adjust": adjust_questions,
        "answers": generate_answers,
        "visualization": visualization_scores,  # two arguments: user_input, ablation (bool)
        "csv": to_csv, # two arguments: user_input, ablation (bool)
        "ablation": ablation_study
    }
    asyncio.get_event_loop().run_until_complete(excute_function[generate_type](user_input))