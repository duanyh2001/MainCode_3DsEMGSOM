"""
File Name: 3D_sEMG_SOM.py
Description: This is a script for gesture classification using the 3D-sEMG-SOM algorithm proposed by the author,
             which includes the following functions:
             -3D-sEMG-SOM design
             -Data processing
             -Training of gesture classification model
             -Result display
Author: Yahan DUAN (duanyh@mial.nankai.edu.cn; 2330126751@qq.com)
Date: 2024-06-13
Version: 1.0
Copyright: Copyright (c) 2024 Yahan DUAN

Revision History:
    2024-06-13 - Yahan DUAN - Initial version
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, scale
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import random
import tqdm
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
from feature_extraction import FeatureExtractor
from util_functions import rms_detect, data_filter
import sklearn.preprocessing as sp
import os
import math
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


class MiniSom3D:
    """
    3D_sEMG_SOM class
    """
    def __init__(self, x, y, z, input_len, sigma=1.0, learning_rate=0.5):
        self._x = x
        self._y = y
        self._z = z
        self._input_len = input_len
        self._sigma = sigma
        self._learning_rate = learning_rate
        self._weights = np.random.rand(x, y, z, input_len)
        self._activation_map = np.zeros((x, y, z))

        # Define neighborhood index
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)
        self._neigz = np.arange(z)

    def _activate(self, x):
        s = np.subtract(x, self._weights)
        it = np.nditer(self._activation_map, flags=['multi_index'])
        for i in it:
            self._activation_map[it.multi_index] = np.linalg.norm(s[it.multi_index])

    def winner(self, x):
        self._activate(x)
        return np.unravel_index(np.argmin(self._activation_map), self._activation_map.shape)

    def _gaussian(self, c, sigma):
        d = 2 * np.pi * sigma ** 2
        ax = np.exp(-np.power(self._neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self._neigy - c[1], 2) / d)
        az = np.exp(-np.power(self._neigz - c[2], 2) / d)
        return np.outer(ax, np.outer(ay, az)).reshape((self._x, self._y, self._z))

    def update(self, x, win, t, max_iter):
        eta = self._learning_rate * np.exp(-t / max_iter)
        sig = self._sigma * np.exp(-t / max_iter)
        g = self._gaussian(win, sig)
        for i in range(self._x):
            for j in range(self._y):
                for k in range(self._z):
                    self._weights[i, j, k] += g[i, j, k] * eta * (x - self._weights[i, j, k])

    def classify(self, x):
        win = self.winner(x)
        return win


def visualize_som_weights(som):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    weights = som._weights
    x, y, z = np.indices(weights.shape[:3])
    ax.scatter(weights[:, :, :, 0].flatten(), weights[:, :, :, 1].flatten(), weights[:, :, :, 2].flatten(), c='b',
               marker='o')

    ax.set_xlabel('Weight X')
    ax.set_ylabel('Weight Y')
    ax.set_zlabel('Weight Z')
    plt.show()


def visualize_classification(som, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    for i, label in enumerate(np.unique(labels)):
        class_data = data[labels == label]
        for point in class_data:
            win = som.winner(point)
            ax.scatter(win[0], win[1], win[2], c=colors[i % len(colors)], marker='o',
                       label=f'Class {label}' if i == 0 else "")

    ax.set_xlabel('Neuron X')
    ax.set_ylabel('Neuron Y')
    ax.set_zlabel('Neuron Z')
    plt.legend()
    plt.show()


# Delete data during rest
def extract_data(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 1) * 200, 0), min((i + 4) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# Delete data during rest (another)
def extract_data_1(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 4) * 200, 0), min((i + 1) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# Delete non switching data: front ->back
def extract_data_change_f(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 2) * 200, 0), min((i + 3) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# Delete non switching data: back ->front
def extract_data_change_b(raw_data, motion_num):
    tensor_list = np.array([])
    # delete_tensor = np.arange(0, 200-50)
    # tensor_list = np.append(tensor_list, delete_tensor)
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        if i == 0:
            delete_tensor = np.arange(0, 3*200)
            tensor_list = np.append(tensor_list, delete_tensor)
        delete_tensor = np.arange(max((i - 3) * 200, 0), min(i * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)
        delete_tensor = np.arange(max(i * 200+200, 0), min((i+3) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


def feature_extract(input_data):
    feature_ex = FeatureExtractor(data=input_data)
    mav_feature = feature_ex.MAV()
    rms_feature = feature_ex.RMS()
    zc_feature = feature_ex.ZC()
    ssc_feature = feature_ex.SSC()
    wl_feature = feature_ex.WL()
    dwt_feature = feature_ex.DWT()
    # dft_feature = feature_ex.DFT()

    return np.hstack([rms_feature, zc_feature])


def read_csv(root_diretory, motion_num):
    data_list = []
    dir_len = len(os.listdir(root_diretory)) - 1
    # print(dir_len)
    # w_num = (1 + np.sqrt(1 + 8 * dir_len)) / 2
    w_num = 4
    csv_num = 6
    print("number of steady motions: ", w_num)
    length = motion_num * 6 + 1
    # Static state action allocation label
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        file_dir = root_dir + csv_name
        emg_df = pd.read_csv(file_dir)
        # emg_df = emg_df.drop(emg_df.index[0:400])
        emg_data = emg_df.values[:(length * 200), :]
        emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
        if i == 0:
            emg_data = extract_data_1(emg_data, motion_num)  # 删除休息状态行
            emg_data = (data_filter(emg_data.T)).T  # 低通滤波
            # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段

            feature_arr = feature_extract(emg_data)
            # print(len(emg_data))
            # feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
            # std_scale = sp.StandardScaler().fit(feature_arr)
            # feature_arr = std_scale.transform(feature_arr)
        else:
            emg_data = extract_data(emg_data, motion_num)  # 删除休息状态行
            emg_data = (data_filter(emg_data.T)).T  # 低通滤波
            # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段

            feature_arr = feature_extract(emg_data)
            # print(len(emg_data))
            # feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
            # std_scale = sp.StandardScaler().fit(feature_arr)
            # feature_arr = std_scale.transform(feature_arr)

        # 转化成DataFrame
        feature_df = pd.DataFrame(feature_arr)
        feature_df['label'] = i
        data_list.append(feature_df)
        # 稳态数据处理完毕判断
        if (i+1) == w_num:
            break

    # Switch state assignment label
    # """
    # 前->后
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        if i >= 1:
            file_dir = root_dir + csv_name
            emg_df = pd.read_csv(file_dir)
            # emg_df = emg_df.drop(emg_df.index[0:400])
            emg_data = emg_df.values[:(length * 200), :]
            emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
            emg_data = extract_data_change_f(emg_data, motion_num)  # 删除非切换状态行
            emg_data = (data_filter(emg_data.T)).T  # 低通滤波
            # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段

            feature_arr = feature_extract(emg_data)
            # print(len(emg_data))
            # feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
            # std_scale = sp.StandardScaler().fit(feature_arr)
            # feature_arr = std_scale.transform(feature_arr)
            # 转化成DataFrame
            feature_df = pd.DataFrame(feature_arr)
            feature_df['label'] = i + w_num - 1
            data_list.append(feature_df)

    # 后->前
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        if i >= 1:
            file_dir = root_dir + csv_name
            emg_df = pd.read_csv(file_dir)
            # emg_df = emg_df.drop(emg_df.index[0:400])
            emg_data = emg_df.values[:(length * 200), :]
            emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
            emg_data = extract_data_change_b(emg_data, motion_num)  # 删除非切换状态行
            emg_data = (data_filter(emg_data.T)).T  # 低通滤波
            # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段

            feature_arr = feature_extract(emg_data)
            # print(feature_arr)
            # feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
            # std_scale = sp.StandardScaler().fit(feature_arr)
            # feature_arr = std_scale.transform(feature_arr)
            # 转化成DataFrame
            feature_df = pd.DataFrame(feature_arr)
            feature_df['label'] = i + w_num - 1 + csv_num - 1
            data_list.append(feature_df)
    # """

    emg_df = pd.concat(data_list, ignore_index=True)
    # emg_df.to_csv('D:/emg_data/continuous/dyh/online_data/test4/result2.csv')
    emg_df = emg_df.reindex(np.random.permutation(emg_df.index))  # 打乱

    X = emg_df.values[:, :-1]
    # X = scale(X)  # zero-mean normalization
    Y = emg_df.values[:, -1]
    return X, Y, emg_df


# Feature standardization (x-mu)/std
def feature_normalization(data):
    mu = np.mean(data, axis=0, keepdims=True)
    sigma = np.std(data, axis=0, keepdims=True)
    return (data - mu) / sigma


if __name__ == '__main__':

    root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/dyh/offline_data/trial4/'
    feature_col = [
        # 'mav_ch1', 'mav_ch2', 'mav_ch3', 'mav_ch4', 'mav_ch5', 'mav_ch6', 'mav_ch7', 'mav_ch8'
        'rms_ch1', 'rms_ch2', 'rms_ch3', 'rms_ch4', 'rms_ch5', 'rms_ch6', 'rms_ch7', 'rms_ch8',
        # 'ssc_ch1', 'ssc_ch2', 'ssc_ch3', 'ssc_ch4', 'ssc_ch5', 'ssc_ch6', 'ssc_ch7', 'ssc_ch8',
        'zc_ch1', 'zc_ch2', 'zc_ch3', 'zc_ch4', 'zc_ch5', 'zc_ch6', 'zc_ch7', 'zc_ch8',
        # 'wl_ch1', 'wl_ch2', 'wl_ch3', 'wl_ch4', 'wl_ch5', 'wl_ch6', 'wl_ch7', 'wl_ch8',
        # 'dwt_ch1', 'dwt_ch2', 'dwt_ch3', 'dwt_ch4', 'dwt_ch5', 'dwt_ch6', 'dwt_ch7', 'dwt_ch8',
    ]
    # MOTION_LIST = {0: '休息', 1: '展开', 2: '握拳', 3: '食指', 4: '休息->展开', 5: '休息->握拳', 6: '休息->食指', 7: '展开->握拳', 8: '展开->食指',
    #                9: '握拳->食指', 10: '展开->休息', 11: '握拳->休息', 12: '食指->休息', 13: '握拳->展开', 14: '食指->展开', 15: '食指->握拳'}
    MOTION_LIST = {0: '休息', 1: '展开', 2: '握拳', 3: '按键', 4: '休息->展开', 5: '休息->握拳', 6: '休息->按键',
                   7: '展开->握拳', 8: '展开->按键',
                   9: '展开->休息', 10: '握拳->休息', 11: '按键->休息', 12: '握拳->展开', 13: '按键->展开'}
    # MOTION_LIST = {0: '休息', 1: '展开', 2: '握拳', 3: '食指'}
    # MOTION_LIST = {4: '休息->展开', 5: '休息->握拳', 6: '休息->食指', 7: '展开->握拳', 8: '展开->食指', 9: '握拳->食指',
    #                10: '展开->休息', 11: '握拳->休息', 12: '食指->休息', 13: '握拳->展开', 14: '食指->展开', 15: '食指->握拳'}
    motion_num = 8
    data, labels, emg_df = read_csv(root_dir, motion_num)
    label_names = MOTION_LIST

    # 对训练数据进行正则化处理
    # datas = normalize(datas, norm='l2')
    # data = feature_normalization(data)

    # data = np.random.rand(100, 10)  # 示例数据
    # labels = np.random.randint(0, 2, size=100)  # 二分类标签

    # # 数据标准化
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)

    N, D = np.shape(data)
    print('shape of train data:', (N, D))

    # 经验公式：决定输出层尺寸
    x = y = z = math.ceil(np.power(5 * np.sqrt(N), 1 / 3))
    # x = y = z = 10

    som = MiniSom3D(x=x, y=y, z=z, input_len=data.shape[1], sigma=1.0, learning_rate=0.5)

    n_iterations = 100
    for t in range(n_iterations):
        idx = np.random.randint(0, len(data))
        som.update(data[idx], som.winner(data[idx]), t, n_iterations)

    visualize_som_weights(som)
    visualize_classification(som, data, labels)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    total_time = 0

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        som = MiniSom3D(x=x, y=y, z=z, input_len=X_train.shape[1], sigma=1.0, learning_rate=0.5)

        n_iterations = 100
        start_time = time.time()
        for t in range(n_iterations):
            idx = np.random.randint(0, len(X_train))
            som.update(X_train[idx], som.winner(X_train[idx]), t, n_iterations)
        train_time = time.time() - start_time

        correct_count = 0
        start_time = time.time()
        for i in range(len(X_test)):
            pred = som.classify(X_test[i])
            pred_label = y_train[np.argmin([np.linalg.norm(X_train[idx] - X_test[i]) for idx in range(len(X_train))])]
            if pred_label == y_test[i]:
                correct_count += 1
        test_time = time.time() - start_time

        accuracy = correct_count / len(X_test)
        accuracies.append(accuracy)
        total_time += (train_time + test_time)

    average_accuracy = np.mean(accuracies)

    print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
    print(f'Total Computation Time: {total_time:.2f} seconds')

