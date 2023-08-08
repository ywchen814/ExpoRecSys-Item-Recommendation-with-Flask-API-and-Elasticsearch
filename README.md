# ExpoRecSys: Item Recommendation with Flask API and Elasticsearch

## Objective

ExpoRecSys is a powerful item recommendation system designed for the dynamic exhibition industry. This GitHub project provides a comprehensive solution that combines Flask API for deploying the recommendation model and Elasticsearch for efficient text data storage and retrieval. With ExpoRecSys, attendees can effortlessly explore and discover products that align with their interests, thereby enhancing their exhibition experience while also boosting manufacturers' revenue.

## Technical Features

- Utilization of **Sentence-BERT** for semantic search and generating similar product recommendations
- Implementation of **Unsupervised Learning** techniques for building effective recommendation models
- **Flask API** for deploying the item-based recommendation model
- **Elasticsearch** integration for efficient text data storage and retrieval

## How to Use ExpoRecSys

1. Pull the Elasticsearch Docker image: `docker pull docker.elastic.co/elasticsearch/elasticsearch:8.8.2`
2. Create a Docker network: `docker network create elastic`
3. Run the Elasticsearch container: `docker run --name es --net elastic -p 9200:9200 -it docker.elastic.co/elasticsearch/elasticsearch:8.8.2`
4. Set the required system configuration: `sudo sysctl -w vm.max_map_count=262144`
5. Start the Elasticsearch container: `docker start es`
6. Install the required Python dependencies: `pip install -r requirements.txt` or `conda env create -f ./environment.yml`
7. Run `api.py` to start the Flask API server: `python api.py`

### Data Preprocessing

Before using ExpoRecSys, you need to preprocess the data into a dictionary format and generate embeddings from text information. Follow these steps:

1. Open `Data-Preprocessing.ipynb`
2. Execute the notebook and load the data to 
    *   preprocess the data and store it into `./data` in dictionary format
    *   generate embeddings using Sentence-BERT and store it into `./embedding`
3. run `es_store_data.py` to load data from `./data` with a dictionary format and store it in Elasticsearch 

## API Documentation

ExpoRecSys API provides various endpoints to interact with the recommendation system and search for products using different methods.

**Base URL:** `http://127.0.0.1:5000/`

**General Notes:**
- Please ensure all data is sent in JSON format for all endpoints.
- The server will respond with JSON data for each endpoint.

### Search Product by ID
#### `/search_by_id` (POST)

**Description:**
Given a product ID, this endpoint searches for the product in Elasticsearch and returns its information.

**Request:**

```json
{
    "Product_id": "CU0103347409"
}
```
**Response:**

```json
{
    "Description": "適合用於修補用途",
    "Description_en": "Suitable to be used in mending purpose mainly in agricultural sector.",
    "Product_Name": "PE 修補膠帶",
    "Product_Name_en": "PE repair tape",
    "Product_id": "CU0004601801"
}
```
### Recommend Products by ID
#### `/recommend_by_id` (POST)

**Description:**
Given a product ID, this endpoint returns the top recommended products along with their similarity scores.

**Request:**

```json
{
    "Product_id": "CU0103347409"
}
```
**Response:**

```json
[
    {
        "Product_id": "CU0004601803",
        "Score": 0.856
    }
]
```
### Recommend Products by Keywords
#### `/recommend_by_keywords` (POST)

**Description:**
Given keywords, this endpoint returns the top recommended products related to the provided keywords, along with their similarity scores.

**Request:**

```json
{
    "Keywords": "tape"
}
```
**Response:**

```json
[
    {
        "Product_id": "CU0004601803",
        "Score": 0.856
    },
    {
        "Product_id": "CU0004601804",
        "Score": 0.492
    }
]
```
### Recommend Products by Keywords using Elasticsearch
#### `recommend_by_keywords_es`  (POST)

**Description:**
Given keywords, this endpoint uses Elasticsearch to search for products related to the keywords and returns the top recommended product IDs.

**Request:**

```json
{
    "Keywords": "tape"
}
```
**Response:**

```json
[
    {
        "Product_id": "CU0004601803"
    },
    {
        "Product_id": "CU0004601804"
    }
]
```
### Note

* For endpoints /recommend_by_id, /recommend_by_keywords, and /recommend_by_keywords_es, the returned products are sorted in descending order of similarity scores.
* For endpoint /search_by_id, if the product with the given ID is not found, an empty response will be returned.
* Please ensure that you send the appropriate JSON request to the respective endpoints and handle the responses accordingly.

## File Structure

- `api.py`: Implement the Flask API with endpoints for item recommendation and product search
- `Data-Preprocessing.ipynb`: Preprocess data into dictionary format and generate embeddings using Sentence-BERT
- `es.py`: Contain functions to connect to Elasticsearch, perform product searches based on keywords
- `es_store_data.py`: Load data from `data/` in a dictionary format and store it in Elasticsearch
- `utils.py`: Provide utility functions for the recommendation model, including Recommending Top-K, Embedding Normalization, Loading Tensor Embeddings
- `data/`: Directory to store data in a dictionary format for immediate storage in Elasticsearch
- `embedding/`: Directory to store product name and description embeddings transformed from Sentence-BERT

## Dataset Attributes

- Product_id: The unique ID of the product.
- Product_Name: The name or title of the product in Chinese.
- Vendor_id: The unique ID of the vendor.
- Main_Category: The main category or exhibition type of the product.
- Sub_Category: The sub-category or specific category of the product.
- Description: The text description of the product in Chinese.
- Product_Name_en: The name or title of the product in English.
- Description_en: The text description of the product in English.

<!-- ## License

ExpoRecSys is licensed under the [MIT License]. -->

## Contact

For any questions or inquiries, feel free to reach out to [grant19990814@gmail.com](grant19990814@gmail.com).

---


