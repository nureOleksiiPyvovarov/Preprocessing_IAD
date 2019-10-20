import numpy as np
import pandas as pd

import math
from sklearn.datasets import load_iris

data = load_iris()
iris_data = data.data

attributes = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'CLASS']


def column_arr(arr):
    col_array = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if j in range(-len(col_array), len(col_array)):
                col_array[j].append(arr[i][j])
            else:
                col_array.insert(j, [arr[i][j]])
    return col_array


def avg(arr):
    summ = 0
    for item in arr:
        summ += item
    return round(summ / len(arr), 6)


def dataFrame(arr, target):
    iris = {}
    c_array = column_arr(arr)
    c_array.append(target)
    for item in range(len(attributes)):
        iris[attributes[item]] = c_array[item][:]
    df = pd.DataFrame(iris, columns=attributes)
    datasets = {}
    by_class = df.groupby('CLASS')
    for groups, data in by_class:
        datasets[groups] = data
    return datasets


def attribute_avg(arr, target):
    datasets = dataFrame(arr, target)
    avg_arr = []
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_avg = []
        for column in datasets[i]:
            columnSeriesObj = datasets[i][column]
            column_avg.append(avg(columnSeriesObj.values))
        avg_arr.append(column_avg)
    return avg_arr


def attribute_med(arr, target):
    datasets = dataFrame(arr, target)
    med_arr = []
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_med = []
        for column in datasets[i]:
            columnSeriesObj = datasets[i][column]
            columnSeriesObj.values.sort()
            item = columnSeriesObj.values
            if len(item) % 2 != 0:
                column_med.append(item[math.floor(len(item) / 2)])
            else:
                avg_med = avg([item[int(len(item) / 2)], item[int(len(item) / 2) + 1]])
                column_med.append(avg_med)
        med_arr.append(column_med)
    return med_arr


def attribute_half_sum(arr, target):
    datasets = dataFrame(arr, target)
    half_sum = []
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_half_sum = []
        for column in datasets[i]:
            columnSeriesObj = datasets[i][column]
            max_value = max(columnSeriesObj.values)
            min_value = min(columnSeriesObj.values)
            column_half_sum.append(avg([min_value, max_value]))
        half_sum.append(column_half_sum)
    return half_sum


def attribute_mean_square(arr, target):
    datasets = dataFrame(arr, target)
    mean_square = []
    avg_arr = attribute_avg(arr, target)
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_mean_square = []
        for ind, column in enumerate(datasets[i].columns):
            columnSeriesObj = datasets[i][column]
            column_sum = 0
            for item in columnSeriesObj.values:
                column_sum += (item - avg_arr[i][ind]) ** 2
            S = np.sqrt(column_sum / len(datasets[i].columns))
            column_mean_square.append(S)
        mean_square.append(column_mean_square)
    return mean_square


def attribute_avg_module(arr, target):
    datasets = dataFrame(arr, target)
    avg_module = []
    med_arr = attribute_med(arr, target)
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_avg_module = []
        for ind, column in enumerate(datasets[i].columns):
            columnSeriesObj = datasets[i][column]
            column_sum = 0
            for item in columnSeriesObj.values:
                column_sum += (item - med_arr[i][ind])
            Delta = np.absolute(column_sum / len(datasets[i].columns))
            column_avg_module.append(Delta)
        avg_module.append(column_avg_module)
    return avg_module


def attribute_swing(arr, target):
    datasets = dataFrame(arr, target)
    swing_arr = []
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop('CLASS', 1)
        column_swing = []
        for column in datasets[i]:
            columnSeriesObj = datasets[i][column]
            max_value = max(columnSeriesObj.values)
            min_value = min(columnSeriesObj.values)
            column_swing.append(max_value - min_value)
        swing_arr.append(column_swing)
    return swing_arr


def attribute_estimation(data):
    for i in range(len(data)):
        data[i] = (list(map(lambda x: x ** 2, data[i])))
    return data


print(attribute_avg(iris_data, data.target))
print(attribute_med(iris_data, data.target))
print(attribute_half_sum(iris_data, data.target))
print(attribute_mean_square(iris_data, data.target))
print(attribute_avg_module(iris_data, data.target))
print(attribute_swing(iris_data, data.target))
print(attribute_estimation(attribute_mean_square(iris_data, data.target)))
