from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# Use environment variable for API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# if not api_key:
#     raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    # repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",

    huggingfacehub_api_token=api_key,
)

result = llm.invoke('who is the president of Pakistan?')

print(result)