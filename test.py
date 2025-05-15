import os
from pinecone import Pinecone

# Load your Pinecone API key (can also set directly)
api_key = os.getenv("PINECONE_API_KEY")  # Ensure you have this set in your environment variables

if api_key is None:
    print("Error: Pinecone API key is missing!")
else:
    try:
        # Initialize the Pinecone instance with the API key
        pc = Pinecone(api_key=api_key)

        # List indexes to check if the connection is working
        index_list = pc.list_indexes().names()
        
        # Check if indexes are available
        if index_list:
            print(f"API key is valid. Here are your indexes: {index_list}")
        else:
            print("API key is valid, but no indexes found.")
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}")
