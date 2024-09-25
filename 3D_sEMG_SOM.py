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
    2024-09-09 - Yahan DUAN - Modify the problem of model prediction
"""
import time

import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
from feature_extraction import FeatureExtractor
from util_functions import rms_detect, data_filter
import sklearn.preprocessing as sp
import os
import math
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, scale


def weights_PCA(X, Y, Z, data):
    N, D = np.shape(data)
    weights = np.zeros([X, Y, Z, D])

    pc_length, pc = np.linalg.eig(np.cov(np.transpose(data)))
    pc_order = np.argsort(-pc_length)
    for i, c1 in enumerate(np.linspace(-1, 1, X)):
        for j, c2 in enumerate(np.linspace(-1, 1, Y)):
            for k, c3 in enumerate(np.linspace(-1, 1, Z)):
                weights[i, j, k] = c1 * pc[pc_order[0]] + c2 * pc[pc_order[1]] + c3 * pc[pc_order[2]]
    return weights


def gaussion_neighborhood(X, Y, Z, c, sigma):
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z))
    d = 2 * sigma * sigma
    ax = np.exp(-np.power(xx - xx.T[c], 2) / d)
    ay = np.exp(-np.power(yy - yy.T[c], 2) / d)
    az = np.exp(-np.power(zz - zz.T[c], 2) / d)
    return (ax * ay * az).T


def bubble_neighborhood(X, Y, Z, c, sigma):
    neigx = np.arange(X)
    neigy = np.arange(Y)
    neigz = np.arange(Z)

    ax = np.zeros([X, Y, Z])
    ay = np.zeros([Y, Z, X])
    az = np.zeros([Z, X, Y])
    for i, c1 in enumerate(neigx):
        for j, c2 in enumerate(neigy):
            for k, c3 in enumerate(neigz):
                ax[i, j, k] = ((neigx[i] > c[0] - sigma) and (neigx[i] < c[0] + sigma)) * 1.
                ay[j, k, i] = ((neigy[j] > c[1] - sigma) and (neigy[j] < c[1] + sigma)) * 1.
                az[k, i, j] = ((neigz[k] > c[2] - sigma) and (neigz[k] < c[2] + sigma)) * 1.
    return (ax * ay * az).T


def get_learning_rate(lr, t, max_steps):
    return lr / (1 + t / (max_steps / 2))


def euclidean_distance(x, w):
    dis = np.expand_dims(x, axis=(0, 1, 2)) - w
    return np.linalg.norm(dis, axis=-1)


def feature_normalization(data):
    mu = np.mean(data, axis=0, keepdims=True)
    sigma = np.std(data, axis=0, keepdims=True)
    return (data - mu) / sigma


def get_winner_index(x, w):
    dis = euclidean_distance(x, w)
    index = np.where(dis == np.min(dis))
    return (index[0][0], index[1][0], index[2][0])


class ThreeDSOM:
    def __init__(self, x, y, z, N_epoch=100, init_lr=2.0, sigma=0.8, init_weight_fun=None, neighborhood_fun=None, seed=20):
        self.x = x
        self.y = y
        self.z = z
        self.N_epoch = N_epoch
        self.init_lr = init_lr
        self.sigma = sigma
        self.init_weight_fun = init_weight_fun if init_weight_fun else weights_PCA
        self.neighborhood_fun = neighborhood_fun if neighborhood_fun else gaussion_neighborhood
        self.seed = seed
        self.weights = None
        self.winmap = None

    def train(self, datas):
        N, D = np.shape(datas)
        N_steps = self.N_epoch * N
        rng = np.random.RandomState(self.seed)
        if self.init_weight_fun is None:
            self.weights = rng.rand(self.x, self.y, self.z, D) * 2 - 1
            self.weights /= np.linalg.norm(self.weights, axis=-1, keepdims=True)
        else:
            self.weights = self.init_weight_fun(self.x, self.y, self.z, datas)

        for n_epoch in range(self.N_epoch):
            index = rng.permutation(np.arange(N))
            for n_step, _id in enumerate(index):
                x = datas[_id]
                t = N * n_epoch + n_step
                eta = get_learning_rate(self.init_lr, t, N_steps)
                winner = get_winner_index(x, self.weights)
                new_sigma = get_learning_rate(self.sigma, t, N_steps)
                g = self.neighborhood_fun(self.x, self.y, self.z, winner, new_sigma)
                g = g * eta
                self.weights = self.weights + np.expand_dims(g, -1) * (x - self.weights)
        return self.weights

    def labels_map(self, data, labels):
        if not len(data) == len(labels):
            raise ValueError('data and labels must have the same length.')
        winmap = defaultdict(list)
        for x, l in zip(data, labels):
            winmap[get_winner_index(x, self.weights)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap

    def classify(self, data, winmap):
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = get_winner_index(d, self.weights)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result


def plot_weights_3d(weights):
    """
    展示输出层权重，将每个节点的权重在3D网格中可视化。
    :param weights: 权重矩阵，形状为 (X, Y, Z, D)，其中 D 是特征维度
    """
    X, Y, Z, D = weights.shape

    # 计算每个节点的权重范数（作为可视化指标）
    weight_norms = np.linalg.norm(weights, axis=-1)

    # 创建3D坐标网格
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z))

    # 3D展示
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 用颜色表示权重范数的大小
    img = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=weight_norms.flatten(), cmap='viridis', s=100)

    # 添加颜色条
    fig.colorbar(img, ax=ax, label='Weight Norm')

    # 设置坐标轴标签
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of SOM Weights')

    # 显示图形
    plt.show()


# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'


# 定义绘制权重热图的函数
def plot_weight_heatmaps(weights):
    """
    绘制输出层权重的热图，将权重在各个2D平面上展示。
    :param weights: 权重矩阵，形状为 (X, Y, Z, D)，其中 D 是特征维度
    """
    X, Y, Z, D = weights.shape

    # 计算每个节点的权重范数（作为热图强度的依据）
    weight_norms = np.linalg.norm(weights, axis=-1)

    # 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 绘制X-Y平面的权重热图
    im1 = axes[0].imshow(np.mean(weight_norms, axis=2), cmap='plasma', interpolation='nearest', aspect='auto')
    axes[0].set_title('Mean Weights in X-Y Plane', fontsize=24)
    axes[0].set_xlabel('X axis', fontsize=24)
    axes[0].set_ylabel('Y axis', fontsize=24)
    axes[0].tick_params(axis='both', which='major', labelsize=16)  # 调整坐标轴刻度字体大小
    axes[0].set_xticks(np.arange(X))  # 设置X轴刻度
    axes[0].set_xticklabels(np.arange(1, X+1))  # 从1开始
    axes[0].set_yticks(np.arange(Y))  # 设置Y轴刻度
    axes[0].set_yticklabels(np.arange(1, Y+1))  # 从1开始

    # 绘制Y-Z平面的权重热图
    im2 = axes[1].imshow(np.mean(weight_norms, axis=0), cmap='plasma', interpolation='nearest', aspect='auto')
    axes[1].set_title('Mean Weights in Y-Z Plane', fontsize=24)
    axes[1].set_xlabel('Y axis', fontsize=24)
    axes[1].set_ylabel('Z axis', fontsize=24)
    axes[1].tick_params(axis='both', which='major', labelsize=16)  # 调整坐标轴刻度字体大小
    axes[1].set_xticks(np.arange(Y))  # 设置Y轴刻度
    axes[1].set_xticklabels(np.arange(1, Y+1))  # 从1开始
    axes[1].set_yticks(np.arange(Z))  # 设置Z轴刻度
    axes[1].set_yticklabels(np.arange(1, Z+1))  # 从1开始

    # 绘制X-Z平面的权重热图
    im3 = axes[2].imshow(np.mean(weight_norms, axis=1), cmap='plasma', interpolation='nearest', aspect='auto')
    axes[2].set_title('Mean Weights in X-Z Plane', fontsize=24)
    axes[2].set_xlabel('X axis', fontsize=24)
    axes[2].set_ylabel('Z axis', fontsize=24)
    axes[2].tick_params(axis='both', which='major', labelsize=16)  # 调整坐标轴刻度字体大小
    axes[2].set_xticks(np.arange(X))  # 设置X轴刻度
    axes[2].set_xticklabels(np.arange(1, X+1))  # 从1开始
    axes[2].set_yticks(np.arange(Z))  # 设置Z轴刻度
    axes[2].set_yticklabels(np.arange(1, Z+1))  # 从1开始

    # 创建一个统一的横向颜色条，并调整其长宽
    cbar = fig.colorbar(im3, ax=axes, orientation='horizontal', shrink=1.0, aspect=40, pad=0.15)
    cbar.set_label('Weight Norms', fontsize=24)  # 设置颜色条的标签字体大小
    cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小

    # 显示图形
    plt.show()




def plot_3d_weights(weights):
    """
    3D散点图展示权重节点的分布
    :param weights: 权重矩阵，形状为 (X, Y, Z, D)，其中 D 是特征维度
    """
    X, Y, Z, D = weights.shape

    # 将每个节点的权重范数作为强度
    weight_norms = np.linalg.norm(weights, axis=-1)

    # 创建坐标网格
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 散点图，使用节点的权重范数作为点的大小
    scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=weight_norms.flatten(), cmap='Blues', s=50)

    # 添加坐标轴标签
    ax.set_xlabel('Weight X')
    ax.set_ylabel('Weight Y')
    ax.set_zlabel('Weight Z')

    # 添加颜色条
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

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

    return np.hstack([mav_feature, rms_feature, zc_feature, ssc_feature])


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


if __name__ == '__main__':

    root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/zjc/offline_data/trial1/'
    feature_col = [
        'mav_ch1', 'mav_ch2', 'mav_ch3', 'mav_ch4', 'mav_ch5', 'mav_ch6', 'mav_ch7', 'mav_ch8'
        'rms_ch1', 'rms_ch2', 'rms_ch3', 'rms_ch4', 'rms_ch5', 'rms_ch6', 'rms_ch7', 'rms_ch8',
        'zc_ch1', 'zc_ch2', 'zc_ch3', 'zc_ch4', 'zc_ch5', 'zc_ch6', 'zc_ch7', 'zc_ch8',
        'ssc_ch1', 'ssc_ch2', 'ssc_ch3', 'ssc_ch4', 'ssc_ch5', 'ssc_ch6', 'ssc_ch7', 'ssc_ch8',
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
    data = normalize(data, norm='l2')

    N, D = np.shape(data)
    print('shape of train data:', (N, D))

    # 经验公式：决定输出层尺寸
    # x = y = z = math.ceil(np.power(5 * np.sqrt(N), 1 / 3)) + 3
    x = y = z = 10
    # x = y = z = 6

    som = ThreeDSOM(x, y, z, N_epoch=100, init_lr=2.0, sigma=0.8, init_weight_fun=weights_PCA,
                    neighborhood_fun=gaussion_neighborhood)

    # 记录训练开始时间
    start_train_time = time.time()
    weights = som.train(data)
    print('shape of weights', np.shape(weights))

    # 生成 labels_map
    winmap = som.labels_map(data, labels)

    # plot_weights_3d(weights)
    plot_weight_heatmaps(weights)
    # plot_3d_weights(weights)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accuracies = []
    times = []

    for fold, (train_index, test_index) in enumerate(kf.split(data), 1):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        # 创建SOM对象
        som = ThreeDSOM(x, y, z, N_epoch=100, init_lr=2.0, sigma=0.8, init_weight_fun=weights_PCA,
              neighborhood_fun=gaussion_neighborhood)

        # 记录训练开始时间
        start_train_time = time.time()
        weights = som.train(X_train)
        print('shape of weights', np.shape(weights))

        # 生成 labels_map
        winmap = som.labels_map(X_train, y_train)

        # 记录训练结束时间
        end_train_time = time.time()

        # 进行分类
        start_test_time = time.time()
        y_pred = som.classify(X_test, winmap)
        end_test_time = time.time()

        # 计算并记录时间
        train_time = end_train_time - start_train_time
        test_time = end_test_time - start_test_time
        times.append(test_time)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # 每折反馈输出
        print(f"Fold {fold} Results:")
        print(f"  - Training time: {train_time:.4f} seconds")
        print(f"  - Testing time: {test_time:.4f} seconds")
        print(f"  - Accuracy: {accuracy*100:.2f}%")
        print("-" * 30)

    average_accuracy = np.mean(accuracies)
    average_test_time = np.mean(times)

    print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
    print(f'Average Test time: {average_test_time*1000:.2f} ms')
