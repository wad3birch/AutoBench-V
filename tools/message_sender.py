import os, sys
import asyncio
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.azure_client import client, config
from openai import AzureOpenAI

def send_message(message, image_url=None):
    client = AzureOpenAI(
        api_key=config['api']['key'],
        api_version=config['api']['version'],
        azure_endpoint=config['api']['endpoint']
    )
    if image_url is None:
        response = client.chat.completions.create(
            model=config['api']['model'],
            messages=[{"role": "user", "content": message}],
        )
    else:
        response = client.chat.completions.create(
            model=config['api']['model'],
            messages=[
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": message
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ] } 
            ],
        )  
    return response.choices[0].message.content

def send_multiple_messages(messages, image_urls=None):
    responses = [None] * len(messages)
    
    def task_wrapper(index, message, image_url=None):
        if image_url is None:
            response = send_message(message)
        else:
            response = send_message(message, image_url)
        responses[index] = response

    with ThreadPoolExecutor(max_workers=5) as executor:
        if image_urls is None:
            futures = [
                executor.submit(task_wrapper, index, message)
                for index, message in enumerate(messages)
            ]
        else:
            futures = [
                executor.submit(task_wrapper, index, message, image_url)
                for index, (message, image_url) in enumerate(zip(messages, image_urls))
            ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

    return responses

async def async_send_message(message, image_url=None):
    if image_url is None:
        response = await client.chat.completions.create(
            model=config['api']['model'],
            messages=[{"role": "user", "content": message}],
        )
    else:
        response = await client.chat.completions.create(
            model=config['api']['model'],
            messages=[
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": message
                    },
                    { 
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ] } 
            ],
        )  
    return response.choices[0].message.content

