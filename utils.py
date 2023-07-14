import numpy as np
import torch
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sentence_transformers.util import semantic_search


model_en = SentenceTransformer('all-mpnet-base-v2')
translator = GoogleTranslator(source='zh-TW', target='english')

def recommend_top_k_by_id(product_id: str, id_embedding_dict: dict, k: int=5, threshold: float = 0.75) -> Tuple[List[str], List[float]]:
    '''
    Recommend the top k similar products given a product ID.

    Args:
        product_id (str): 
            The ID of the product for which recommendations will be generated.

        id_embedding_dict (dict): 
            A dictionary mapping product IDs (str) to their corresponding embeddings.

        k (int): 
            The number of top similar products to recommend.

        threshold (float): 
            The threshold value for similarity scores. 
            Only recommendations with scores equal to or greater than this threshold will be considered.

    Returns:
        tuple:
            A tuple containing two lists:
            - The recommended product IDs.
            - The corresponding similarity scores.

    Example:
        rec_pd_ids, rec_scores = recommend_top_k('CU0004601801', id_embedding_dict, 5, 0.75)
    '''
    item_embedding = id_embedding_dict[product_id]
    products_id = list(id_embedding_dict.keys())
    items_embedding = list(id_embedding_dict.values())
    # Perform semantic search to find similar items
    rec_k_dic = semantic_search(item_embedding, items_embedding, top_k=k+1)[0] 
    # Drop the given item itself
    rec_k_dic = np.delete(rec_k_dic, 0)
    indices = []
    rec_score_ls = []
    for item in rec_k_dic:
        score = round(item['score'], 3)
        if score >= threshold:
            indices.append(item['corpus_id'])
            rec_score_ls.append(round(item['score'], 3)) 

    # Retrieve the recommended product IDs using the indices
    rec_pd_id_ls = [products_id[index] for index in indices]
    
    return (rec_pd_id_ls, rec_score_ls)


def recommend_top_k_by_keywords(keywords: str, words_embedding_dict: dict, k: int=5, threshold: float = 0.0) -> Tuple[List[str], List[float]]:
    '''
    Recommend the top k similar products given keywords.

    Args:
        keywords (str): 
            The keywords used for generating recommendations.

        words_embedding_dict (dict): 
            A dictionary mapping product IDs (str) to their corresponding embeddings.

        k (int): 
            The number of top similar products to recommend.

        threshold (float): 
            The threshold value for similarity scores. 
            Only recommendations with scores equal to or greater than this threshold will be considered.

    Returns:
        tuple:
            A tuple containing two lists:
            - The recommended product IDs.
            - The corresponding similarity scores.

    Example:
        rec_pd_ids, rec_scores = recommend_top_k('Knife', words_embedding_dict, 5, 0.75)
    '''
    # translate to English
    en_text = translator.translate(keywords)
    print(keywords)
    word_embedding = normalize_embedding(model_en.encode(en_text))
    # Replicate to correspond the shape of words_embedding(product_em_en, des_em_en)
    word_embedding = np.concatenate((word_embedding, word_embedding), axis = 0)  
    products_id = list(words_embedding_dict.keys())
    words_embedding = list(words_embedding_dict.values())

    # Perform semantic search to find similar items
    rec_k_dic = semantic_search(word_embedding, words_embedding, top_k=k)[0] 
    indices = []
    rec_score_ls = []
    for item in rec_k_dic:
        score = round(item['score'], 3)
        if score >= threshold:
            indices.append(item['corpus_id'])
            rec_score_ls.append(round(item['score'], 3)) 

    # Retrieve the recommended product IDs using the indices
    rec_pd_id_ls = [products_id[index] for index in indices]
    
    return (rec_pd_id_ls, rec_score_ls)

def normalize_embedding(embedding, norm=2):
    '''
    Normalize the input embedding vector.

    Args:
        embedding (numpy.ndarray): 
            The input embedding vector to be normalized.
        
        norm (int, default=2): 
            The order of the norm to be applied for normalization.
        
    Returns:
        numpy.ndarray:
            The normalized embedding vector.
    '''
    embedding = embedding / np.linalg.norm(embedding, ord=norm)
    return embedding

def load_tensor_embedding(file_name):
    '''
    Load the dictionary where the key is the product ID and the value is the embedding (as a NumPy array). 
    Convert the embedding into a PyTorch tensor.

    Args:
        file_name (str): 
            The name of the file containing the embedding tensor.

    Returns:
        dict:
            A dictionary mapping product IDs to embedding tensors.

    Example:
        embedding_dict = load_embedding_tensor("item_embedding.pkl")
    '''
    with open(file_name, "rb") as file:
        id_embedding_dict = pickle.load(file)

    keys = id_embedding_dict.keys()
    embeddings = list(id_embedding_dict.values())
    embedding = torch.tensor(embeddings)
    id_embedding_dict = dict(zip(keys, embedding))
    return id_embedding_dict
    