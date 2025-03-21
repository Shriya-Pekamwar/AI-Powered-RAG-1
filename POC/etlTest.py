import os
import openai
import pinecone
from openai import Embedding

# Initialize the Pinecone instance
from pinecone import Pinecone

# Replace these with your actual Pinecone API Key and Index Name
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-index-name"

# Set OpenAI API Key
OPENAI_API_KEY = "your-openai-api-key"
openai.api_key = OPENAI_API_KEY

# Define the OpenAI embedding model
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def get_embedding(text):
    """
    Generate the embedding vector for the input text using OpenAI's embeddings API.
    
    Args:
        text (str): The input text.
        
    Returns:
        list: The embedding vector for the input text.
    """
    try:
        # Use OpenAI's new API to create an embedding
        response = openai.Embedding.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        # Extract and return the embedding vector from the response
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def retrieve_relevant_chunks(year, quarter, query):
    """
    Retrieve relevant document chunks from Pinecone based on a query.
    
    Args:
        year (str): The year of interest.
        quarter (str): The quarter of interest.
        query (str): The search query.
        
    Returns:
        list: Relevant document chunks.
        list: Metadata related to the chunks.
        list: Distances between the chunks and the query.
    """
    # Retrieve the embeddings for the query
    query_embedding = get_embedding(query)
    if query_embedding is None:
        raise ValueError("Failed to generate query embedding")
    
    # Initialize Pinecone index
    index = pc.index(PINECONE_INDEX_NAME)
    
    # Query the index for relevant results based on the query embedding
    try:
        results = index.query(
            vector=query_embedding,
            top_k=5,  # Set the number of relevant results you want to retrieve
            include=["documents", "metadatas", "distances"],
            filter={
                "year": year,
                "quarter": quarter
            }
        )
    except Exception as e:
        print(f"Error during Pinecone query: {e}")
        return [], [], []
    
    # Extract results and their metadata
    chunks = results['matches']
    metadatas = [match['metadata'] for match in chunks]
    distances = [match['score'] for match in chunks]
    
    return chunks, metadatas, distances

def generate_response(query, chunks, metadatas, distances):
    """
    Generate a response using OpenAI's GPT model based on the retrieved chunks.
    
    Args:
        query (str): The user's question.
        chunks (list): The relevant document chunks retrieved from Pinecone.
        metadatas (list): Metadata associated with the chunks.
        distances (list): The distances between the query and retrieved chunks.
        
    Returns:
        str: The generated response.
    """
    # Combine chunks with their metadata for better context
    context_with_sources = []
    for i, (chunk, metadata, distance) in enumerate(zip(chunks, metadatas, distances)):
        context_with_sources.append(f"Source [{i+1}] ({metadata.get('source', 'Unknown')}): {chunk['text']}")
    
    context = "\n\n".join(context_with_sources)
    
    # Define system message for GPT model
    system_message = """You are a helpful assistant that answers questions based on the given context. 
    If the context doesn't contain relevant information, acknowledge that and provide general information."""
    
    # Define user message with query and context
    user_message = f"""Question: {query}
    
    Context information:
    {context}
    
    Please answer the question based on the context provided."""
    
    try:
        # Use OpenAI's GPT model to generate a response based on the context
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can use a different model like "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,  # Adjust temperature for more focused answers
            max_tokens=1000   # Limit response length
        )
        
        # Return the generated response text
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

def process_query(year, quarter, query):
    """
    Process the user's query, retrieve relevant chunks, and generate an answer using the LLM.
    
    Args:
        year (str): The year of interest.
        quarter (str): The quarter of interest.
        query (str): The user's question.
        
    Returns:
        str: The generated answer to the query.
    """
    # Retrieve relevant chunks based on the query
    chunks, metadatas, distances = retrieve_relevant_chunks(year, quarter, query)
    
    if not chunks:
        return "Sorry, no relevant information found for the specified year and quarter."
    
    # Generate response using OpenAI's LLM
    response = generate_response(query, chunks, metadatas, distances)
    return response

if __name__ == "__main__":
    # Example query
    year = "2024"
    quarter = "third"
    query = "What are the financial results for the third quarter of fiscal 2024?"
    
    # Process the query and generate an answer
    response = process_query(year, quarter, query)
    
    print(f"Answer: {response}")
