import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from utils import write_answer_list
from config import Config
from Week2.Arrays import actual_0, predicted_0, actual_1, predicted_1
from Week2.Arrays import actual_2, predicted_2, actual_0r, predicted_0r
from Week2.Arrays import actual_1r, predicted_1r, actual_10, predicted_10
from Week2.Arrays import actual_11, predicted_11


ANSWER_PATH = Config.get_answer_path('Week_2', 'task_2')


def plot_f_metric(actual, predicted):
    thresholds = np.arange(0, 10, 1)
    f1_values = list()
    for t in thresholds:
        f1 = f1_score(actual, predicted > t*0.1)
        f1_values.append(f1)
    plt.figure(figsize=(12, 7))
    plt.title('F1')
    plt.plot(
        thresholds,
        f1_values,
        linestyle='-',
        color='blue',
    )
    plt.xlabel('Thresholds')
    plt.ylabel('F1')
    plt.show()
    index_max_f1 = f1_values.index(max(f1_values))
    return index_max_f1


def weighted_log_loss(actual, predicted, weight):
    return - np.mean(
        weight * actual * np.log(predicted) +
        (1 - weight) * (1 - actual) * np.log(1 - predicted)
    )


def get_min_distant(actual, predicted):
    initial = np.array([0, 1])
    destinations = list()
    fpr, tpr, thr = roc_curve(actual, predicted)
    for i, j in zip(fpr, tpr):
        point = np.array([i, j])
        destinations.append(np.linalg.norm(point - initial))
    min_destination = np.min(destinations)
    min_destinations_indexes = np.argwhere(destinations == min_destination)
    answer = max(thr[min_destinations_indexes])
    return answer[0]


def task_1():
    threshold = 0.65
    precisions = list()
    recalls = list()
    answer = list()
    for actual, predicted in zip(
            [actual_1, actual_10, actual_11],
            [predicted_1 > threshold, predicted_10 > threshold, predicted_11 > threshold]
    ):
        precisions.append(precision_score(actual, predicted))
        recalls.append(recall_score(actual, predicted))
    for precision, recall in zip(precisions, recalls):
        answer.append(precision)
        answer.append(recall)
    return answer


def task_2():
    thresholds = list()
    for actual, predicted in zip(
            [actual_1, actual_10, actual_11],
            [predicted_1, predicted_10, predicted_11]
    ):
        max_index = plot_f_metric(actual, predicted)
        thresholds.append(max_index)
    return thresholds


def task_3():
    answers = list()
    weight = 0.3
    for actual, predicted in zip([actual_0, actual_1, actual_2, actual_0r, actual_1r, actual_10, actual_11],
                                 [predicted_0, predicted_1, predicted_2, predicted_0r, predicted_1r, predicted_10, predicted_11]):
        answers.append(weighted_log_loss(actual, predicted, weight))
    return answers


def task_4():
    answer = list()
    for actual, predicted in zip([actual_0, actual_1, actual_2, actual_0r, actual_1r, actual_10, actual_11],
                                 [predicted_0, predicted_1, predicted_2, predicted_0r, predicted_1r, predicted_10,
                                  predicted_11]):
        answer.append(get_min_distant(actual, predicted))
    return answer


def main():
    write_answer_list(task_1(), os.path.join(ANSWER_PATH, 'task_1.txt'))
    write_answer_list(task_2(), os.path.join(ANSWER_PATH, 'task_2.txt'))
    write_answer_list(task_3(), os.path.join(ANSWER_PATH, 'task_3.txt'))
    write_answer_list(task_4(), os.path.join(ANSWER_PATH, 'task_4.txt'))
