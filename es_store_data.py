from elasticsearch import Elasticsearch
import pickle

ELASTIC_PASSWORD = "vL0pU8d_kLmcXUzrT0rT"

# Connect to Elasticsearch
es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="./http_ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# Load the data
file_name = './Data/df_dict.pkl'
with open(file_name, "rb") as file:
    df_dict = pickle.load(file)

# Store the data in Elasticsearch
index_name = 'products'
for i, product in enumerate(df_dict):
    es.index(index=index_name, id = product['Product_id'], body=product)

count_query = {
    "query": {
        "match_all": {}
    }
}

# Perform the count query
response = es.count(index=index_name, body=count_query)

# Extract the total count of documents
total_documents = response['count']

print(f"Total number of documents in the index '{index_name}': {total_documents}")