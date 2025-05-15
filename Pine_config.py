import os
from pinecone import Pinecone, ServerlessSpec

# Optional: set via environment variable
# os.environ["PINECONE_API_KEY"] = "your-key-here"

# Initialize Pinecone client
pc = Pinecone(api_key="pcsk_2co5j_TnMqjQoRNTXZSg9jnnCkgPSdYCus9agEJ43ncb2fLWn463JGbcscgjN8jFm4fVx")

index_name = "bamboo"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to index
pinecone_index = pc.Index(index_name)

# Confirm success
print("Pinecone index ready:", pinecone_index.describe_index_stats())
