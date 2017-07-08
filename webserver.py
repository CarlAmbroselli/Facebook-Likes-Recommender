from flask import Flask, jsonify, request
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
        recommendation=[{'value': x[0], 'score' :x[1]} for x in results if str(x[0]) != 'nan']
    )

@app.route('/likes', methods=['GET'])
def get_likes():
    return jsonify(like_ids())

@app.route('/details/<id>', methods=['GET'])
def get_details(id):
    return like_id_to_item(int(id)).to_json()

@app.route('/search/<text>', methods=['GET'])
def search_data(text):
    return search(text).sort_values('talking_about_count', ascending=False).head(20).to_json(orient='records')

@app.route('/similar/<id>', methods=['GET'])
def get_similar(id):
    return jsonify(
        similar=similar_items(int(id))
    )

app.run(debug=True)
# Run interactive shell afterwards
# code.interact(local=locals())