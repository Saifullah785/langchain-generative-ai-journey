
# Import the HuggingFaceEmbeddings class for creating embeddings
# Import load_dotenv to load environment variables from a .env file
# Import the os module for interacting with the operating system
# Import numpy for numerical operations (although not explicitly used in the current version)
# Import cosine_similarity to calculate the similarity between vectors

from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv 
import os  
import numpy as np  
from sklearn.metrics.pairwise import cosine_similarity  

 # Load environment variables from the .env file (e.g., API keys)

load_dotenv() 

 # Retrieve the Hugging Face API token from the environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN") 

# Initialize HuggingFaceEmbeddings with a specific model
embeddings = HuggingFaceEmbeddings(

    # Specify the model name for generating embeddings (a smaller, faster model)
    model_name="sentence-transformers/all-MiniLM-L6-v2",  

    # huggingfacehub_api_token=HUGGINGFACE_API_KEY #Uncomment this line if the model requires an API key
)

# Define a list of documents (in this case, short descriptions of cricketers)
documents = ["Imran Khan: A charismatic leader and all-rounder.",
             "Wasim Akram: The 'Sultan of Swing,' a bowling maestro.",
             "Waqar Younis: Fearsome pace and reverse swing king.",
             "Inzamam-ul-Haq: A prolific, powerful, and sometimes slow batsman.",
             "Babar Azam: Modern master, elegant and consistent scorer."]

 # Define the query to find relevant documents

query = "Who is the babar azam?" 

# Generate embeddings for the documents

doc_embeddings = embeddings.embed_documents(documents)

# Generate an embedding for the query

query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity between the query embedding and each document embedding

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the index of the document with the highest similarity score

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]  # Sort by score and get the last (highest) value

# Print the query, the most similar document, and the similarity score
print(query)  # Print the original query
print(documents[index])  # Print the document that is most similar to the query
print('similarity score:', score)  # Print the cosine similarity score between the query and the most similar