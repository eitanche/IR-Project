import matplotlib.pyplot  as plt
import random
import json

TRAIN_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29]
TEST_IDS =[0, 7, 11, 16, 26]
EXPERIMENTS_FILE= "only_indexes_merged_weights_precision_scores_two_word_index_precision_new.json"


def choose_test_queries():
    query_id = list(range(0,30))
    chosen_ids = []
    for i in range(5):
        chosen_one = random.choice(query_id)
        query_id.remove(chosen_one)
        chosen_ids.append(chosen_one)
    print(chosen_ids)

def organize_test_and_train():
    test_query_ids = sorted([26, 7, 0, 11, 16])
    train_query_ids = list(range(0,30))
    for test_id in test_query_ids:
        train_query_ids.remove(test_id)

    print(train_query_ids)
    print(test_query_ids)

def find_average_train_test_for_each_option():
    with open(EXPERIMENTS_FILE,"r") as f:
        weigths_precision_dict = json.load(f)

    weight_for_each_index = dict()
    for w_s, precision_list in weigths_precision_dict.items():
        train_average = calculate_average(precision_list, TRAIN_IDS)
        test_average = calculate_average(precision_list, TEST_IDS)
        #w_s = "title wight:0.0,body wight:0.0,anchor wight:1.0"
        list_of_all_weights = w_s.split(",")
        all_weights = [float(string_weight.split(':')[1]) for string_weight in list_of_all_weights]
        title_weight, body_weight, anchor_weight = tuple(all_weights)
        weight_for_each_index[w_s]={"title_weight":title_weight,"body_weight":body_weight,"anchor_weight":anchor_weight,"train":train_average, "test":test_average}
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "w") as f:
        json.dump(weight_for_each_index, f)

def calculate_average(precision_list, ids):
    sum_of_precisions = 0
    for query_id in ids:
        sum_of_precisions+=precision_list[query_id]
    return sum_of_precisions/len(ids)

def choose_maximum_by_train():
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "r") as f:
        k_b_to_dict = json.load(f)

    best_k_b = sorted([(k_b, {"train":result_dict["train"], "test":result_dict["test"]}) for k_b, result_dict in k_b_to_dict.items()], key=lambda x:x[1]["train"], reverse=True)[:40]
    for item in best_k_b:
        print(item)

find_average_train_test_for_each_option()
choose_maximum_by_train()


def print_file():
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "r") as f:
        k_b_to_dict = json.load(f)
        best_20_train = sorted([(dict['title_weight'],dict['body_weight'],dict['anchor_weight'],dict['train'],dict['test']) for dict in k_b_to_dict.values()],key=lambda x: x[3],reverse=True)[:20]
        for scores in best_20_train:
            print(scores)


# print_file()# print_file()

