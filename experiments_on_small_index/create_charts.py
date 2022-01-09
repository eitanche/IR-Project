import json
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = "find_index_weights/two_word/only_indexes_merged_weights_precision_scores_two_word_index_averages.json"
K=20

def chart_plot():
    labels, train, test = create_np_matrix_of_results()

    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(labels))
    train_chart = ax.bar(x - width/2, train, width, label='Train')
    test_chart = ax.bar(x + width/2, test, width, label='Test')


    ax.set_ylabel('MAP@40 Score')
    ax.set_title('MAP@40 Score of Index Weight Configurations')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(train_chart, padding=3)
    ax.bar_label(test_chart, padding=3)

    fig.tight_layout()

    plt.show()

def create_np_matrix_of_results():
    list_of_result_dictionaries = get_K_top_results_by_average_precision()
    #zip* list op tuples --> 2 lists
    labels ,train, test =  zip(*[(get_result_dictionary_string(result_dictionary), round(result_dictionary["train"],3), round(result_dictionary["test"],3))  for result_dictionary in list_of_result_dictionaries])
    return labels,np.array(train),np.array(test)

def get_K_top_results_by_average_precision():
    with open(FILE_NAME,"r") as f:
        result_json = json.load(f)
    return sorted(result_json.values(), key=lambda x:x.get("train")+x.get("test"), reverse=True)[:K]

def get_result_dictionary_string(result_dictionary):
    return f"title: {round(result_dictionary['title_weight'],2)}, body: {round(result_dictionary['body_weight'],2)}, anchor: {round(result_dictionary['anchor_weight'],2)}"

chart_plot()