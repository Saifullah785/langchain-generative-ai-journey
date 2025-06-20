# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

# Load environment variables from .env file (for API keys, etc.)
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Define the schema for the structured output using TypedDict and type annotations
class Review(TypedDict):
    # List of key themes discussed in the review
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    # Brief summary of the review
    summary: Annotated[str, "A brief summary of the review"]
    # Sentiment of the review: positive or negative
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    # List of pros mentioned in the review
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    # List of cons mentioned in the review
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    # Name of the reviewer
    name: Annotated[Optional[str], "Write the name of the reviewer"]
    
# Create a structured model that will return output matching the Review schema
structured_model = model.with_structured_output(Review)

# Invoke the model with a sample review text and get the structured result
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by saifullah
""")

# Print the name of the reviewer from the result
print(result["name"])