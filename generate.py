import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def generate_response(retrieved_texts, query, max_tokens=150):
    """
    Generates a response based on the retrieved texts and query using Gemini.

    Args:
    retrieved_texts (list): List of retrieved text strings.
    query (str): Query string.
    max_tokens (int): Maximum number of tokens for the response.

    Returns:
    str: Generated response.
    """
    # Combine retrieved texts and query into a prompt
    context = "\n".join(retrieved_texts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Generate response using Gemini
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    return response.text
