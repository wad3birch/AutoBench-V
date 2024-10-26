# azure_client.py
from config.configuration import config
from openai import AzureOpenAI, AsyncAzureOpenAI

# Initialize AzureOpenAI client
client = AsyncAzureOpenAI(
    api_key=config['api']['key'],
    api_version=config['api']['version'],
    azure_endpoint=config['api']['endpoint']
)
