# ExpoRecSys: Item Recommendation with Flask API and Elasticsearch

## Introduction

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
5. Start the Elasticsearch container: `docker start es`.
6. Install the required Python dependencies: `pip install -r requirements.txt`
7. Run `api.py` to start the Flask API server: `python api.py`

## API Endpoints

### `/search_by_id` (POST)

Given a product ID, this endpoint searches for the product in Elasticsearch and returns its information.

### `/recommend_by_id` (POST)

Given a product ID, this endpoint returns the top recommended products along with their similarity scores.

### `/recommend_by_keywords` (POST)

Given keywords, this endpoint returns the top recommended products related to the provided keywords, along with their similarity scores.

### `/recommend_by_keywords_es` (POST)

Given keywords, this endpoint uses Elasticsearch to search for products related to the keywords and returns the top recommended product IDs.

## File Structure

- `api.py`: Implements the Flask API with endpoints for item recommendation and product search
- `es.py`: Contains functions to connect to Elasticsearch, perform product searches based on keywords
- `utils.py`: Provides utility functions for the recommendation model, including Recommending Top-K, Embedding Normalization, Loading Tensor Embeddings
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


