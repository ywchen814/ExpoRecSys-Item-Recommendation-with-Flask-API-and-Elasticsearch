from flask import Flask, request, jsonify
from utils import *

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to Recommendation System!'

@app.route('/recommend_by_id', methods=['POST'])
def recommend_by_id():
    if request.json:
        product_id = request.json['Product_id']
    rec_pd_id_ls, rec_score_ls = recommend_top_k_by_id(product_id, id_embedding_dict)
    response = []

    for p_id, p_score in zip(rec_pd_id_ls, rec_score_ls):
        dictionary = {"Product_id": p_id, 'Score':p_score}
        response.append(dictionary)

    return jsonify(response)

@app.route('/recommend_by_keywords', methods=['POST'])
def recommend_by_keywords():
    keywords = request.json['Keywords']
    rec_pd_id_ls, rec_score_ls = recommend_top_k_by_keywords(keywords, words_embedding_dict)
    response = []

    for p_id, p_score in zip(rec_pd_id_ls, rec_score_ls):
        dictionary = {"Product_id": p_id, 'Score':p_score}
        response.append(dictionary)

    return jsonify(response)

if __name__ == '__main__':

    id_embedding_dict = load_tensor_embedding('.\embedding\item_embedding.pkl')
    words_embedding_dict = load_tensor_embedding('.\embedding\words_embedding.pkl')
    app.run(host='127.0.0.1', port=5000, debug=True)
