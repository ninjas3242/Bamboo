from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# AWS Region
region = "us-east-1"

# Get AWS Credentials
session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'aoss', session_token=credentials.token)

# OpenSearch Serverless Endpoint
host = "https://34b3vjfw9q910bdo9jx5.us-east-1.aoss.amazonaws.com"

# Initialize OpenSearch Client
client = OpenSearch(
    hosts=[host],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Test connection
try:
    print("OpenSearch Info:", client.info())
except Exception as e:
    print("Failed to connect to OpenSearch:", e)

# Index Name
INDEX_NAME = "documents"

# Index Configuration
index_body = {
    "settings": {
        "index.knn": True
    },
    "mappings": {
        "properties": {
            "embeddings": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "engine": "faiss",
                    "name": "hnsw",
                    "space_type": "l2"
                }
            }
        }
    }
}

# Create Index if it does not exist
try:
    if not client.indices.exists(INDEX_NAME):
        response = client.indices.create(index=INDEX_NAME, body=index_body)
        print("✅ Index created:", response)
    else:
        print("ℹ️ Index already exists")
except Exception as e:
    print("❌ Error creating index:", e)
