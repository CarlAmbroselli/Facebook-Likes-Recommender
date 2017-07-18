from flask import Flask, jsonify, request
from json import load
from prediction import *
from flask_cors import CORS, cross_origin
import math
import code

app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/recommend', methods=['POST'])
def do_recommendation():
    data = request.get_json(silent=True)
    if not 'likes' in data:
        return jsonify(
            error="Please send like ids via JSON like: { likes: [24382, 15489] }"
        )
    results = text_recommendation(data['likes'])
    return jsonify(
        recommendation=[{'value': x[0].to_json(orient='records'), 'score' :x[1]} for x in results if str(x[0].iloc[0]['name']) != 'nan']
    )

@app.route('/likes', methods=['GET'])
def get_likes():
    return jsonify(like_ids())

@app.route('/details/<id>', methods=['GET'])
def get_details(id):
    item = like_id_to_item(int(id))
    item.dataset_like_count = likes_count(item.like_id)
    return item.to_json()

@app.route('/search/<text>', methods=['GET'])
def search_data(text):
    result = search(text)
    result['like_count'] = result['like_id'].apply(lambda x: likes_count(x))
    result = result.sort_values(['like_count', 'talking_about_count'], ascending=False).head(20)
    return result.to_json(orient='records')

@app.route('/explain', methods=['POST'])
def get_explain():
    data = request.get_json(silent=True)
    if not 'likes' in data or not 'id' in data:
        return jsonify(
            error="Please send like ids via JSON like: { id: 132, likes: [24382, 15489] }"
        )
    result = explain(data['id'], data['likes'])
    print(result[1])
    return jsonify(
        total_score=float(result[0]),
        top_contributions=[{
            'like_id': int(model_id_to_like_id(x[0])),
            'contribution': float(x[1]),
            'element': like_id_to_item(int(model_id_to_like_id(x[0]))).to_json(orient='records')
        } for x in result[1]]
    )
@app.route('/similar/<id>', methods=['GET'])
def get_similar(id):
    return jsonify(
        similar=similar_items(int(id))
    )

app.run(debug=True, port=5000)
# Run interactive shell afterwards
# code.interact(local=locals())
