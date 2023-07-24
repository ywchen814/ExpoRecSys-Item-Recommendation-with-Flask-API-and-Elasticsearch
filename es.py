from typing import List, Dict
from elasticsearch import Elasticsearch

def connect_to_elasticsearch(host="localhost", port=9200):
    """
    Connects to Elasticsearch and returns the Elasticsearch object.

    Parameters:
        host (str): The Elasticsearch host URL. Default is "localhost".
        port (int): The Elasticsearch port number. Default is 9200.

    Returns:
        Elasticsearch: The Elasticsearch object on successful connection.
    """
    try:
        es = Elasticsearch(f"http://{host}:{port}")

    # Check if the connection was successful by pinging the cluster
        if es.ping():
            print("Connected to Elasticsearch!")  # Print message if the connection is successful
            return es
        else:
            print("Unable to connect to Elasticsearch.")  # Print message if the connection failed
    except Exception as e:
        print(f"An error occurred: {e}")  # Print any exception that occurred during the connection attempt

ELASTIC_PASSWORD = "RgPL5UEnWEbIr+9oJ8J-"
es = Elasticsearch(
    "https://localhost:9200",
    ca_certs="./http_ca.crt",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)
index_name = 'products'

def recommend_top_k_by_keywords_es(keywords: str, num_results: int = 5) -> List[Dict[str, str]]:
    '''
    Recommend products based on provided keywords using Elasticsearch.

    Args:
        keywords (str):
            The keywords to use for the product search.

        num_results (int, optional):
            The number of recommended products to return. Default is 5.

    Returns:
        List[str]:
            A lists contains the recommended product IDs.

    Example:
        recommendations = recommend_top_k_by_keywords_es("laptop", num_results=10)
    '''

    # Use Elasticsearch to search for products based on the provided keywords
    res = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": keywords,
                    "fields": ['Product_Name', 'Product_Name_en', 'Description', 'Description_en'],
                    "type": "cross_fields",
                    "operator": "or"
                }
            },
            "size": num_results
        }
    )

    rec_pd_id_ls = []
    for hit in res['hits']['hits']:
        rec_pd_id_ls.append(hit['_source']['Product_id'])

    return rec_pd_id_ls

def search_product_by_id(p_id: str) -> dict:
    '''
    Search for a product in Elasticsearch using its ID.

    Args:
        p_id (str):
            The ID of the product to search for in Elasticsearch.

    Returns:
        dict or None:
            If the product is found, returns a dictionary containing the product information.
            If the product is not found or an error occurs during the search, returns None.

    Example:
        product_info = search_products_by_id("CU0004601801")
    '''
    try:
        result = es.get(index=index_name, id=p_id)
        return result['_source']
    except Exception as e:
        print(f"Error: {e}")
        return None