# coding: utf-8

import pandas as pd
import numpy as np
import implicit
from scipy.sparse import coo_matrix
import code


def load_data(nrows=5000):
    if nrows == -1:
        like_data = pd.read_csv('./data/user-likes-sorted-clean.csv', sep=',', header=None)
    else:
        like_data = pd.read_csv('./data/user-likes-sorted-clean.csv', sep=',', header=None, nrows=nrows)
    data = pd.DataFrame()
    data['user'] = like_data[0].astype("category")
    data['like'] = like_data[1].astype("category")
    return data

def load_test_data(nrows=5496):
    full_file_len = 95045419 #take only the last nrows rows
    skiprows = full_file_len-nrows
    like_data = pd.read_csv('./data/user-likes-sorted-clean.csv', sep=',', header=None, nrows=nrows, skiprows=skiprows)
    t_data = pd.DataFrame()
    t_data['user'] = like_data[0].astype("category")
    t_data['like'] = like_data[1].astype("category")
    return t_data

def load_mapping():
    mapping_data = pd.read_csv('./data/mapping.csv', sep=',')
    return mapping_data

def run_test_one_vs_all():
    grouped = test_data.groupby('user')
    test_results = []
    progress = 0
    max_progress = len(grouped)
    for name, group in grouped:
        likes = group['like'].values
        user_results = []
        for index in range(min(10, len(likes))):
            filtered_likes = np.concatenate((np.array(likes)[:index], np.array(likes)[index + 1:]))
            search_like = likes[index]
            # print("Calculate", index, search_like, "using # likes:", len(filtered_likes))
            result = explain(search_like, filtered_likes)
            # print("Result", result[0])
            user_results.append(result[0])
        test_results.append(np.average(user_results))
        progress += 1
        if (progress % 10 == 0):
            print("Progress: ", progress, "/", max_progress)
            print("Current result:", np.average(test_results))
    return np.average(test_results)

def run_test_fast():
    grouped = test_data.groupby('user')
    test_results = []
    # progress = 0
    # max_progress = len(grouped)
    for name, group in grouped:
        likes = group['like'].values
        user_results = []
        if (len(likes) > 100):
            # print("Test user", len(likes))
            filtered_likes = np.array(likes)[11:]
            search_likes = likes[:10]
            user_weights = None
            for index in range(10):
                # print("Calculate", index, search_like, "using # likes:", len(filtered_likes))
                result = explain(search_likes[index], filtered_likes, user_weights=user_weights)
                user_weights = result[2]
                # print("Result", result[0])
                user_results.append(result[0])
            test_results.append(np.average(user_results))
        # progress += 1
        # if (progress % 10 == 0):
        #     print("Progress: ", progress, "/", max_progress)
        #     print("Current result:", np.average(test_results))
    return np.average(test_results)


def learn():
    likes = coo_matrix((np.ones(data.shape[0]),
                   (data['like'].cat.codes.copy(),
                    data['user'].cat.codes.copy())))

    likes = np.multiply(likes.todense(), np.ones((likes.shape[0], likes.shape[1]))*40)
    likes = coo_matrix(likes)

    # train model
    model = implicit.als.AlternatingLeastSquares(factors=100)

    # Change to multiplication by inverse logarithm of like count
    model.fit(likes)
    
    return model

def explain(id, likes, user_weights=None):
    return model.explain(userid=0, user_items=user_likes(likes), itemid=like_id_to_model_id(id), user_weights=user_weights)

def similar_items(id=6478112671):
    model_id = like_id_to_model_id(id)
    print(model_id)
    return [(int(model_id_to_like_id(x)), float(y)) for x, y in model.similar_items(model_id)]

def search(text):
    return mapping_data[mapping_data['name'].str.contains(text, na=False)]
    # return mapping_data[mapping_data['name'].str.contains(text) | mapping_data['category'].str.contains(text)
    #                     | mapping_data['category_list'].str.contains(text)
    #                     | mapping_data['genre'].str.contains(text)]

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

def likes_count(like):
    try:
        return like_counts.iloc[like_counts.keys().get_loc(like) - 1]
    except KeyError:
        return 0

def perform_recommendation(likes):
    return model.recommend(userid=0, user_items=user_likes(likes), recalculate_user=True, N=20)

def text_recommendation(likes):
    return list(map(lambda x: (like_id_to_item(model_id_to_like_id(x[0])), int(round(x[1]*1000))/1000), perform_recommendation(likes)))

def test(training_size=5000, test_size=5496):
    global data
    global test_data
    global model
    data = load_data(training_size)
    # test_data = load_test_data(test_size)
    model = learn()
    test_result = run_test_fast()
    f = open('results.txt', 'a')
    f.write("Test for training size: " + str(training_size) +
          "\tTest size: " + str(test_size) +
          "\tResult: " + str(test_result) + '\n')
    f.close()
    print("Test for training size:", training_size,
          "Test size:", test_size,
          "Result:", test_result)
    return test_result

def test_loop():
    i = 1024
    test_size = 5496
    global test_data
    test_data = load_test_data(test_size)
    while i < 2 ** 27:
        test(training_size=i)
        i *= 2

confidence = 40
data = load_data(5000)
test_data = load_test_data(1000)
model = learn()
like_counts = data['like'].value_counts()
# print(like_counts.head())
mapping_data = load_mapping()

# slipknot = 6478112671
# similar_items(slipknot)

# recommendation = text_recommendation([24382, 15489])
# print(recommendation)

code.interact(local=locals())