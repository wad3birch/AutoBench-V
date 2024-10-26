import asyncio
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.lvm_pool import async_gpt4o


config_file_path = './config/config.yaml'
with open(config_file_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

align_question_template = config.get('tifa_question_prompt')
align_answer_template = config.get('tifa_answer_prompt')

async def generate_and_evaluate_image(image_path, prompt):
    description = prompt

    prompt = align_question_template.format(description=description)
    response_text = await async_gpt4o(prompt)

    questions_results = parse_response_text(response_text, description, image_path)
    
    tasks = [
        answer_question(question, align_answer_template)
        for question in questions_results
    ]
    responses = await asyncio.gather(*tasks)
    
    correct = sum(1 for r in responses if r['llm_answer'] == r['correct_answer'])
    total = len(responses)
    score = correct / total if total > 0 else 0

    results = {
        'image_path': image_path,
        'score': score,
        'responses': responses
    }

    return float(score), results

def parse_response_text(response_text, description, image_path):
    questions_results = []
    for question_block in response_text.split('\n\n'):
        if not question_block.strip():
            continue

        element = {}
        in_choices = False
        choices_list = []
        for line in question_block.split('\n'):
            if line.startswith('question:'):
                element['question'] = line[len('question:'):].strip()
            elif line.startswith('choices:'):
                in_choices = True
                choices_str = line[len('choices:'):].strip()
                if choices_str.startswith('['):
                    choices_str = choices_str[1:]
                if choices_str.endswith(']'):
                    choices_str = choices_str[:-1]
                if choices_str:
                    choices_list.extend([choice.strip().strip('"') for choice in choices_str.split(',') if choice.strip()])
            elif in_choices:
                if line.strip().endswith(']'):
                    in_choices = False
                    line = line.strip()[:-1]
                if line.startswith('answer:') or line.startswith('element_type:') or line.startswith('element:'):
                    in_choices = False
                else:
                    choices_list.extend([choice.strip().strip('"') for choice in line.strip().split(',') if choice.strip()])
            if not in_choices:
                if line.startswith('answer:'):
                    element['answer'] = line[len('answer:'):].strip()
                elif line.startswith('element_type:'):
                    element['element_type'] = line[len('element_type:'):].strip()
                elif line.startswith('element:'):
                    element['element'] = line[len('element:'):].strip()

        if choices_list:
            element['choices'] = choices_list

        if 'question' in element:
            questions_results.append({
                'caption': description,
                'image_path': image_path,
                'question': element.get('question'),
                'choices': element.get('choices'),
                'answer': element.get('answer'),
                'element_type': element.get('element_type'),
                'element': element.get('element'),
            })
    return questions_results

async def answer_question(question, align_answer_template):
    prompt = align_answer_template.format(
        question=question['question'],
        choices=question['choices']
    )
    response = await async_gpt4o(prompt, question['image_path'])
    
    # Extract the answer from the response
    answer_prefix = "answer: "
    llm_answer = response.strip()
    if llm_answer.startswith(answer_prefix):
        llm_answer = llm_answer[len(answer_prefix):].strip()

    llm_response = {
        'question': question['question'],
        'choices': question['choices'],
        'correct_answer': question['answer'],
        'llm_answer': llm_answer,
        'element_type': question['element_type'],
        'element': question['element'],
        'image_path': question['image_path']
    }
        
    return llm_response

# if __name__ == '__main__':
#     image_path = "D:\\Paper\\visual_autobench\\code\\document\\spatial_understanding\\extracted_images\\medium\\5fa81c01-c665-4687-9422-caec8a2cba62.png"
#     prompt = "A cat sitting on a table in a kitchen with various utensils hanging on the wall behind it."
#     results = asyncio.get_event_loop().run_until_complete(generate_and_evaluate_image(image_path, prompt))
#     print(results)
