import replicate
import sys
import os
import asyncio
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)
from config.azure_client import config
import json
from openai import AsyncAzureOpenAI, AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# os.environ['http_proxy'] = os.getenv("HTTP_PROXY")
# os.environ['https_proxy'] = os.getenv("HTTPS_PROXY")
# os.environ['REPLICATE_API_TOKEN'] = os.getenv("REPLICATE_API_TOKEN")

def sdxl(prompt):
    output = replicate.run(
    "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    input={
        "width": 512,
        "height": 512,
        "prompt": prompt,
        "refine": "expert_ensemble_refiner",
        "scheduler": "K_EULER",
        "lora_scale": 0.6,
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "apply_watermark": False,
        "high_noise_frac": 0.8,
        "negative_prompt": "",
        "prompt_strength": 0.8,
        "num_inference_steps": 25
    }
    )
    return output

def stable_diffusion(prompt):
    output = replicate.run(
    "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
    input={
        "width": 512,
        "height": 512,
        "prompt": prompt,
        "scheduler": "K_EULER",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
    )
    return output

def openjourney_v4(prompt):
    output = replicate.run(
    "prompthero/openjourney-v4:e8818682e72a8b25895c7d90e889b712b6edfc5151f145e3606f21c1e85c65bf",
    input={
        "seed": 3329637825,
        "width": 512,
        "height": 768,
        "prompt": prompt,
        "scheduler": "K_EULER_ANCESTRAL",
        "num_outputs": 1,
        "guidance_scale": 7,
        "negative_prompt": "bad anatomy, blurry, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated",
        "prompt_strength": 0.8,
        "num_inference_steps": 25
    }
    )
    return output

def stable_diffusion_3(prompt):
    output = replicate.run(
    "stability-ai/stable-diffusion-3",
    input={
        "cfg": 3.5,
        "steps": 28,
        "prompt": prompt,
        "aspect_ratio": "3:2",
        "output_format": "webp",
        "output_quality": 90,
        "negative_prompt": "",
        "prompt_strength": 0.85
    }
    )
    return output

def dalle_3(prompt):
    client = AzureOpenAI(
        api_key=config['api']['key'],
        api_version=config['api']['version'],
        azure_endpoint=config['api']['endpoint']
    )
    result = client.images.generate(
        model=config['dalle_model'],
        prompt=prompt,
        n=1
    )
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]

    return image_url

async def async_sdxl(prompt):
    output = await replicate.async_run(
        "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
        input={
            "width": 512,
            "height": 512,
            "prompt": prompt,
            "refine": "expert_ensemble_refiner",
            "scheduler": "K_EULER",
            "lora_scale": 0.6,
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "apply_watermark": False,
            "high_noise_frac": 0.8,
            "negative_prompt": "",
            "prompt_strength": 0.8,
            "num_inference_steps": 25
        }
    )
    return output

async def async_stable_diffusion(prompt):
    output = await replicate.async_run(
        "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
        input={
            "width": 512,
            "height": 512,
            "prompt": prompt,
            "scheduler": "K_EULER",
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
    )
    return output

async def async_openjourney_v4(prompt):
    output = await replicate.async_run(
        "prompthero/openjourney-v4:e8818682e72a8b25895c7d90e889b712b6edfc5151f145e3606f21c1e85c65bf",
        input={
            "seed": 3329637825,
            "width": 512,
            "height": 768,
            "prompt": prompt,
            "scheduler": "K_EULER_ANCESTRAL",
            "num_outputs": 1,
            "guidance_scale": 7,
            "negative_prompt": "bad anatomy, blurry, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated",
            "prompt_strength": 0.8,
            "num_inference_steps": 25
        }
    )
    return output

async def async_stable_diffusion_3(prompt):
    output = await replicate.async_run(
        "stability-ai/stable-diffusion-3",
        input={
            "cfg": 3.5,
            "steps": 28,
            "prompt": prompt,
            "aspect_ratio": "3:2",
            "output_format": "webp",
            "output_quality": 90,
            "negative_prompt": "",
            "prompt_strength": 0.85
        }
    )
    return output

async def async_dalle_3(prompt):
    client = AsyncAzureOpenAI(
        api_key=config['api']['key'],
        api_version=config['api']['version'],
        azure_endpoint=config['api']['endpoint']
    )
    result = await client.images.generate(
        model=config['dalle_model'],
        prompt=prompt,
        n=1
    )
    json_response = json.loads(result.model_dump_json())
    image_url = json_response["data"][0]["url"]

    return image_url

async def async_flux_pro(prompt):
    output = await replicate.async_run(
        "black-forest-labs/flux-pro",
        input={
            "steps": 25,
            "prompt": prompt,
            "guidance": 2,
            "interval": 2,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "safety_tolerance": 2
        }
    )
    return output

# async def main():
#     prompt = "A busy city street at dawn, with a woman holding an umbrella walking on the pavement. Nearby, a car drives through a deep puddle, causing a large splash of water to drench the woman. The splashing water is mid-air, reflecting the early morning light, and the woman's expression is one of surprise and irritation. The scene also includes a street vendor setting up a stall, adding to the dynamic environment but keeping the focus on the primary action and reaction."
#     print(await async_flux_pro(prompt))

# if __name__ == "__main__":
#     asyncio.get_event_loop().run_until_complete(main())                  

