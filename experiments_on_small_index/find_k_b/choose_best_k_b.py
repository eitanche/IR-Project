import matplotlib.pyplot  as plt
import random
import json

TRAIN_IDS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29]
TEST_IDS =[0, 7, 11, 16, 26]
EXPERIMENTS_FILE= "all_scores_of_query_expansion.json"


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
        k_b_to_precision_list = json.load(f)

    k_b_to_test_train = dict()
    for k_b, precision_list in k_b_to_precision_list.items():
        train_average = calculate_average(precision_list, TRAIN_IDS)
        test_average = calculate_average(precision_list, TEST_IDS)
        k_b_to_test_train[k_b]={"score":k_b,"train":train_average, "test":test_average}
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "w") as f:
        json.dump(k_b_to_test_train, f)

def calculate_average(precision_list, ids):
    sum_of_precisions = 0
    for query_id in ids:
        sum_of_precisions+=precision_list[query_id]
    return sum_of_precisions/len(ids)

#anchor test 0.7405490012345025 0.1,0.2
#anchor train 0.5456874271635672
def choose_maximum_by_train():
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "r") as f:
        k_b_to_dict = json.load(f)

    best_k_b = sorted([(k_b, {"train":result_dict["train"], "test":result_dict["test"]}) for k_b, result_dict in k_b_to_dict.items() if result_dict["test"]>0.52], key=lambda x:x[1]["train"], reverse=True)[:20]
    for item in best_k_b:
        print(item)

find_average_train_test_for_each_option()
choose_maximum_by_train()


def print_file():
    with open(f"{EXPERIMENTS_FILE[:-5]}_averages.json", "r") as f:
        k_b_to_dict = json.load(f)
        best_20_train = sorted([(dict['k'],dict['b'],dict['train'],dict['test']) for dict in k_b_to_dict.values()],key=lambda x: x[2],reverse=True)[:20]
        for scores in best_20_train:
            print(scores)


# print_file()# print_file()

