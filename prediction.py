# coding: utf-8

import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix
import code


def load_data(nrows=5000):
    like_data = pd.read_csv('./data/user-likes-sorted-clean.csv', sep=',', header=None, nrows=nrows)
    data = pd.DataFrame()
    data['user'] = like_data[0].astype("category")
    data['like'] = like_data[1].astype("category")
    return data

def load_mapping():
    mapping_data = pd.read_csv('./data/mapping.csv', sep=',')
    return mapping_data

def learn():
    likes = coo_matrix((np.ones(data.shape[0]),
                   (data['like'].cat.codes.copy(),
                    data['user'].cat.codes.copy())))
    
    # train model
    model = implicit.als.AlternatingLeastSquares(factors=50)

    # Change to multiplication by inverse logarithm of like count
    model.fit(confidence * likes)
    
    return model

def similar_items(id=6478112671):
    model_id = like_id_to_model_id(id)
    print(model_id)
    return [(int(model_id_to_like_id(x)), float(y)) for x, y in model.similar_items(model_id)]

def like_ids():
    like_objects = dict(enumerate(data['like'].cat.categories))
    like_ids = [int(value) for key, value in like_objects.items()]
    return like_ids

def model_id_to_like_id(id):
    like_objects = dict(enumerate(data['like'].cat.categories))
    return like_objects[id]

def like_id_to_model_id(id):
    like_objects = dict(enumerate(data['like'].cat.categories))
    like_ids = [key for key, value in like_objects.items() if value == id]
    if len(like_ids) > 0:
        return like_ids[0]
    return -1

def like_id_to_item(id):
    return mapping_data[mapping_data['like_id'] == id]

def like_id_to_name(id):
    return mapping_data[mapping_data['like_id'] == id].reset_index()['name'].tolist()[0]

def model_id_to_name(id):
    return like_id_to_name(model_id_to_like_id(id))

def user_likes(likes):
    like_ids = [like_id_to_model_id(like) for like in likes]
    like_ids = [x for x in like_ids if x > 0]
    data = [confidence for _ in like_ids]
    rows = [0 for _ in like_ids]
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((data, (rows, like_ids)), shape=shape).tocsr()

def perform_recommendation(likes):
    return model.recommend(userid=0, user_items=user_likes(likes), recalculate_user=True)

def text_recommendation(likes):
    return list(map(lambda x: (model_id_to_name(x[0]), int(round(x[1]*1000))/1000), perform_recommendation(likes)))

confidence = 40
data = load_data(5000)
model = learn()
mapping_data = load_mapping()

# slipknot = 6478112671
# similar_items(slipknot)

# recommendation = text_recommendation([24382, 15489])
# print(recommendation)
