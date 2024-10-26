import sys
import os
import io, asyncio
from PIL import Image
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
import base64
import anthropic
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import as_completed
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI
load_dotenv()

# os.environ['http_proxy'] = os.getenv("HTTP_PROXY")
# os.environ['https_proxy'] = os.getenv("HTTPS_PROXY")

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def gpt4o(message):
    client = AzureOpenAI(
        api_key=os.getenv("Azure_API_KEY"),
        api_version=os.getenv("Azure_API_VERSION"),
        azure_endpoint=os.getenv("Azure_ENDPOINT")
    )
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": message}],
    )
    return response.choices[0].message.content

async def async_gpt4o(message, image_path=None):
    client = AsyncAzureOpenAI(
        api_key=os.getenv("Azure_API_KEY"),
        api_version=os.getenv("Azure_API_VERSION"),
        azure_endpoint=os.getenv("Azure_ENDPOINT")
    )
    if image_path is None:
        response = await client.chat.completions.create(
            model='gpt-4o',
            messages=[{"role": "user", "content": message}],
        )
    else:
        image = Image.open(image_path)
        base64_image = encode_image_to_base64(image)
        image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        response = await client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        image_message
                    ]
                }
            ],
        )
    return response.choices[0].message.content

async def async_gpt4o_mini(message, image_path=None):
    client = AsyncAzureOpenAI(
        api_key=os.getenv("Azure_API_KEY"),
        api_version=os.getenv("Azure_API_VERSION"),
        azure_endpoint=os.getenv("Azure_ENDPOINT"),
    )
    if image_path is None:
        response = await client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{"role": "user", "content": message}],
        )
        print()
    else:
        image = Image.open(image_path)
        base64_image = encode_image_to_base64(image)
        image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        response = await client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        image_message
                    ]
                }
            ],
        )
    return response.choices[0].message.content

async def async_claude_3_5_sonnet(prompt, image_path=None):
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if image_path is None:
        message = await client.messages.create(
            model='claude-3-5-sonnet-20240620',
            temperature=0.6,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        image_data = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        media_type = "image/webp" 
        message = await client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
    return message.content[0].text

async def async_claude_3_haiku(prompt, image_path=None):
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if image_path is None:
        message = await client.messages.create(
            model='claude-3-haiku-20240307',
            temperature=0.6,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")
        media_type = "image/webp"
        message = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
    return message.content[0].text

async def async_gemini_1_5_flash(prompt, image_path=None):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    if image_path is None:
        response = await model.generate_content_async([prompt])
    else:
        image = Image.open(image_path)
        response = await model.generate_content_async([prompt, image])
        
    return response.text

# async def async_gemini_1_5_pro_exp(prompt, image_path=None):
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0801")
#     if image_path is None:
#         response = await model.generate_content_async([prompt])
#     else:
#         image = Image.open(image_path)
#         response = await model.generate_content_async([prompt, image])
        
#     return response.text

async def async_qwen_vl(prompt, image_path):
    client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    image = Image.open(image_path)
    base64_image = encode_image_to_base64(image)
    image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    response = await client.chat.completions.create(
        model="qwen-vl-max-0809",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_message
                ]
            }
        ],
    )
    return response.choices[0].message.content

async def async_glm_4v(prompt, image_path=None):
    client = AsyncOpenAI(
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )
    if image_path is None:
        response = await client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    image = Image.open(image_path)
    base64_image = encode_image_to_base64(image)
    image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    response = await client.chat.completions.create(
        model="glm-4v",
        messages=[
            {   
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_message
                ]
            }
        ],
    )
    return response.choices[0].message.content

async def async_openai(query, image_file):
    data = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        "model": "gpt-4o",
    }
    if image_file:
        image = Image.open(image_file)
        base64_image = encode_image_to_base64(image)
        image_message = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        data['messages'][0]["content"].append(image_message)

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('api_key')}"
    }
    url = os.getenv('base_url') 
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"Error making request: {e}")
    try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing response: {e}")
        print(response.text)
        return ""
    

# async def main():
#     prompt = """
#   In order to test your ability with pictures, we have a question about Foreground vs. Background area. Please answerbased on your knowledge in this area and your understanding of pictures.
#   Given the image below, answer the questions: Which of the following details is unique to the background and not present in the foreground of the image?\n{\"A\": \"Ornate, shining armor with golden engravings\", \"B\": \"Massive sword wielded by a warrior\", \"C\": \"Indistinct figures of soldiers clashing\", \"D\": \"Fierce eyes and battle scars\"} based on the image. If you can't see the image and can't answer the question, please output "I can't answer this question."
#   Please give the final answer strictly follow the format [[A]] (Srtictly add [[ ]] to the choice, and the content in the brackets should be the choice such as A, B, C, D) and provide a brief explanation of your answer. Directly output your answer in this format and give a brief explanation."""
#     image_path = "D:\\Paper\\visual_autobench\\code\\document\\spatial_understanding\\extracted_images\\hard\\bac2f690-adb7-4b74-9418-3317b75d041c.png"

#     response = await async_claude_3_haiku(prompt)
#     print(response)

# if __name__ == "__main__":
#     asyncio.get_event_loop().run_until_complete(main())