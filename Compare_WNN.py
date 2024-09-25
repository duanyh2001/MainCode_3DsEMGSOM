"""
File Name: Compare_WNN.py
Description: This is a script for gesture classification using the compared method WNN algorithm,
             which includes the following functions:
             -WNN design
             -Data processing
             -Training of gesture classification model
             -Result display
Author: Yahan DUAN (duanyh@mial.nankai.edu.cn; 2330126751@qq.com)
Date: 2024-06-13
Version: 1.0
Copyright: Copyright (c) 2024 Yahan DUAN

Revision History:
    2024-06-13 - Yahan DUAN - Initial version
    2024-09-13 - Yahan DUAN - Update the net
"""

import os
import time
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import warnings
from feature_extraction import FeatureExtractor  # 假定你已经实现了此模块
from util_functions import rms_detect, data_filter  # 假定你已经实现了此模块

# 数据路径和参数
ROOT_DIR = 'D:/MyDocuments/GraduationDesign/data/continuous/dyh/offline_data/trial4/'
MOTION_NUM = 8  # 动作数量
FEATURE_COL = [
    'rms_ch1', 'rms_ch2', 'rms_ch3', 'rms_ch4', 'rms_ch5', 'rms_ch6', 'rms_ch7', 'rms_ch8',
    'zc_ch1', 'zc_ch2', 'zc_ch3', 'zc_ch4', 'zc_ch5', 'zc_ch6', 'zc_ch7', 'zc_ch8'
]


class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db1', level=1):
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        batch_size, features = x.shape
        transformed_list = []

        for i in range(batch_size):
            # 转换为 numpy 数组进行小波变换
            coeffs = pywt.wavedec(x[i].cpu().numpy(), wavelet=self.wavelet, level=self.level)
            coeffs_flattened = np.concatenate([c.flatten() for c in coeffs], axis=0)
            transformed_list.append(coeffs_flattened)

        # 将列表转换为张量
        max_length = max(len(t) for t in transformed_list)  # 获取最大的特征维度
        padded_transformed = np.zeros((batch_size, max_length), dtype=np.float32)

        for i, t in enumerate(transformed_list):
            padded_transformed[i, :len(t)] = t  # 将变换结果填充到张量中

        return torch.tensor(padded_transformed)

class WaveletNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WaveletNet, self).__init__()
        self.wavelet_transform = WaveletTransform(wavelet='db1', level=1)
        # 这里的 input_dim 需要设置为小波变换后的特征维度
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.wavelet_transform(x)
        x = x.view(x.size(0), -1)  # 展平以适应全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def feature_normalization(data):
    """
    特征标准化
    :param data: 原始数据
    :return: 标准化后的数据
    """
    mu = np.mean(data, axis=0, keepdims=True)
    sigma = np.std(data, axis=0, keepdims=True)
    return (data - mu) / sigma


# 删除休息时的数据
def extract_data(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 1) * 200, 0), min((i + 4) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# 删除休息时的数据
def extract_data_1(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 4) * 200, 0), min((i + 1) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# 删除非切换状态的数据：前->后
def extract_data_change_f(raw_data, motion_num):
    tensor_list = np.array([])
    length = motion_num * 6 + 1
    for i in range(0, length, 6):
        delete_tensor = np.arange(max((i - 2) * 200, 0), min((i + 3) * 200, len(raw_data)))
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = tensor_list.astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


# 删除非切换状态的数据：后->前
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
    # 稳定状态动作分配标签
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        file_dir = ROOT_DIR + csv_name
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

    # 切换状态分配标签
    # """
    # 前->后
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        if i >= 1:
            file_dir = ROOT_DIR + csv_name
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
            file_dir = ROOT_DIR + csv_name
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


def main():
    warnings.filterwarnings("ignore")

    # 记录总开始时间
    total_start_time = time.time()

    # 读取并处理数据
    data_start_time = time.time()
    train_set, label, emg_df = read_csv(ROOT_DIR, MOTION_NUM)
    train_set = normalize(train_set, norm='l2')
    train_set = feature_normalization(train_set)
    data_end_time = time.time()
    print(f"数据读取和处理时间: {data_end_time - data_start_time:.2f}秒")

    # K折交叉验证
    kfold_start_time = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    accuracy_list = []
    prediction_time_list = []  # 存储每次预测阶段的时间

    for fold, (train_index, test_index) in enumerate(kf.split(train_set), 1):
        fold_start_time = time.time()

        train_x, train_y = train_set[train_index], label[train_index]
        test_x, test_y = train_set[test_index], label[test_index]

        X_train = torch.tensor(train_x, dtype=torch.float32)
        y_train = torch.tensor(train_y, dtype=torch.long)
        X_test = torch.tensor(test_x, dtype=torch.float32)
        y_test = torch.tensor(test_y, dtype=torch.long)

        # 创建和训练小波神经网络模型
        model = WaveletNet(input_dim=32, output_dim=14)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        epochs = 100

        for epoch in range(epochs):
            epoch_start_time = time.time()

            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            epoch_end_time = time.time()
            print(
                f'Fold {fold} Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Epoch Time: {epoch_end_time - epoch_start_time:.2f}秒')

        # 预测阶段时间计算
        prediction_start_time = time.time()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            total = y_test.size(0)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / total
            accuracy_list.append(accuracy)
        prediction_end_time = time.time()
        prediction_time = prediction_end_time - prediction_start_time
        prediction_time_list.append(prediction_time)  # 记录预测时间

        fold_end_time = time.time()
        print(
            f'Fold {fold} Accuracy: {accuracy:.4f}, Fold Time: {fold_end_time - fold_start_time:.2f}秒, Prediction Time: {prediction_time:.2f}秒')

    kfold_end_time = time.time()
    avg_accuracy = np.mean(accuracy_list)
    avg_prediction_time = np.mean(prediction_time_list)  # 计算平均预测时间

    print(f'Average Accuracy: {avg_accuracy*100:.2f}%')
    print(f'Average Prediction Time: {avg_prediction_time*1000:.2f}ms')  # 输出平均预测时间
    # print(f"K折交叉验证时间: {kfold_end_time - kfold_start_time:.2f}秒")
    #
    # # 记录总结束时间
    # total_end_time = time.time()
    # print(f"总运行时间: {total_end_time - total_start_time:.2f}秒")


if __name__ == "__main__":
    main()
