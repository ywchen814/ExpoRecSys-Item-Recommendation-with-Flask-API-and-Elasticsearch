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

es = connect_to_elasticsearch()

def recommend_top_k_by_keywords_es(keywords: str, num_results: int = 5) -> List[Dict[str, str]]:
    '''
    Recommend products based on provided keywords using Elasticsearch.

    Args:
        keywords (str):
            The keywords to use for the product search.

        num_results (int, optional):
            The number of recommended products to return. Default is 5.

    Returns:
        List[Dict[str, str]]:
            A list of dictionaries representing the recommended products.
            Each dictionary contains the product information, such as Product_Name, Product_Name_en, Description, etc.

    Example:
        recommendations = recommend_top_k_by_keywords_es("laptop", num_results=10)
    '''

    # Use Elasticsearch to search for products based on the provided keywords
    res = es.search(
        index="products",
        body={
            "query": {
                "multi_match": {
                    "query": keywords,
                    "fields": ['Product_Name', 'Product_Name_en', 'Description', 'Description_en'],
                    "type": "cross_fields",
                    "operator": "and"
                }
            },
            "size": num_results
        }
    )

    recommendations = []
    for hit in res['hits']['hits']:
        recommendations.append(hit['_source'])

    return recommendations