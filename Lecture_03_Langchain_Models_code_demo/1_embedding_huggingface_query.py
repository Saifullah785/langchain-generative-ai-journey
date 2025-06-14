from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Use environment variable for API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # huggingfacehub_api_token=api_key,
)

text = "This is a sample sentence."
embedding = embeddings.embed_query(text)

print(len(embedding))
# print(embedding[:5])  # Print the first 5 elements of the embedding
print(embedding)  # Print the full embedding vector