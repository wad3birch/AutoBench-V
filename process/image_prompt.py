import json
import networkx as nx
import yaml
import sys
import os
import asyncio
import aiofiles
import csv
import math
import matplotlib.pyplot as plt

parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from tools.message_sender import async_send_message

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

async def save_csv(data, file_path):
    # Use synchronous file operation for writing CSV
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Topic Word', 'Degree'])
        for topic_word, degree in data:
            writer.writerow([topic_word, degree])

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
            prompt_response = await async_send_message(image_description)
            # print(prompt_response)

            prompt = None
            topic_word = None
            key_words = None
            # prompt_lines = prompt_response.split('\n')
            retry_times = 3
            try:
                prompt_lines = prompt_response.split('\n')
            except:
                for i in range(retry_times):
                    prompt_response = await async_send_message(image_description)
                    prompt_lines = prompt_response.split('\n')
                    if prompt_lines:
                        break

            for line in prompt_lines:
                if line.startswith("Prompt:"):
                    prompt = line[len("Prompt:"):].strip()
                # if line.startswith("Explanation:"):
                #     explanation = line[len("Explanation:"):].strip()
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
                    # "explanation": explanation,
                    # "optimized_prompt": optimize_prompt,
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

                top_nodes = [node for node, degree in sorted(degree_dict.items(), key=lambda item: item[1], reverse=True)[:math.floor(math.log(round_num+1)+1)]]
                used_words.update(top_nodes)

                # Add the topic word and its degree to the list
                all_topic_word_degrees.append((topic_word, degree_dict[topic_word]))

                # Print the top nodes selected in this round
                print(f"Round {round_num + 1} - Top node(s) selected: {top_nodes}")

        # Print the degree of all nodes for this aspect
        print(f"Final degrees for aspect '{aspect}': {dict(G.degree())}")

    return prompts, all_topic_word_degrees

async def main(level):
    # Load configuration
    config_file_path = './config/config.yaml'
    config = await load_config(config_file_path)

    image_prompt_template = config.get('difficulty_control_image_prompt')
    # optimize_prompt_template = config.get('optimized_prompt')
    questions_per_aspect = config.get('questions_per_aspect')
    print(questions_per_aspect)
    user_input = config.get('user_input')
    print(user_input)

    # Load aspects from JSON file
    aspects_file_path = f'./document/{user_input}/{user_input}_guidance.json'
    aspects_data = await load_json(aspects_file_path)

    # Extract aspect names and introductions
    aspects = [(aspect_data['aspect'], aspect_data['introduction'], aspect_data['guidance']) for aspect_data in aspects_data]

    # Generate prompts and collect topic word degrees
    generated_prompts, topic_word_degrees = await generate_prompt_with_topic_words(aspects, image_prompt_template, level, questions_per_aspect)

    # Save prompts to a JSON file
    prompts_file_path = f'./document/{user_input}/{level}_basic_image_prompts.json'
    await save_json(generated_prompts, prompts_file_path)

    # Save topic word degrees to a CSV file
    csv_file_path = f'./document/{user_input}/{level}_topic_word_degrees.csv'
    await save_csv(topic_word_degrees, csv_file_path)

    print("Prompts generated and saved")

if __name__ == '__main__':
    for level in ['hard']:
        asyncio.get_event_loop().run_until_complete(main(level))
