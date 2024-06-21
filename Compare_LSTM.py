"""
File Name: Compare_LSTM.py
Description: This is a script for gesture classification using the compared method LSTM algorithm,
             which includes the following functions:
             -LSTM design
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

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, KFold

from feature_extraction import FeatureExtractor
from util_functions import rms_detect, data_filter


root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/dy/offline_data/trial1/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


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
    # 稳定状态动作分配标签
    # for i, csv_name in enumerate(os.listdir(root_diretory)):
    #     file_dir = root_dir + csv_name
    #     emg_df = pd.read_csv(file_dir)
    #     # emg_df = emg_df.drop(emg_df.index[0:400])
    #     emg_data = emg_df.values[:(length * 200), :]
    #     emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
    #     emg_data = extract_data(emg_data, motion_num)  # 删除休息状态行
    #     emg_data = (data_filter(emg_data.T)).T  # 低通滤波
    #     # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段
    #
    #     feature_arr = feature_extract(emg_data)
    #     # print(len(emg_data))
    #     # feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
    #     # std_scale = sp.StandardScaler().fit(feature_arr)
    #     # feature_arr = std_scale.transform(feature_arr)
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

    # 切换状态分配标签
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


def main():
    total_start_time = time.time()

    root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/dyh/offline_data/trial4/'
    motion_num = 8
    input_size = 16
    hidden_size = 256
    num_layers = 2
    num_classes = 14
    num_epochs = 100
    learning_rate = 0.05
    k_folds = 5
    batch_size = 10

    data_start_time = time.time()
    train_set, label, _ = read_csv(root_dir, motion_num)
    train_set = normalize(train_set, norm='l2')
    data_end_time = time.time()

    print(f"Data Loading and Preprocessing Time: {data_end_time - data_start_time:.2f} seconds")

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=0)
    accuracy_list = []
    kfold_start_time = time.time()

    for fold, (train_index, test_index) in enumerate(kfold.split(train_set), 1):
        fold_start_time = time.time()

        train_x, train_y = train_set[train_index], label[train_index]
        test_x, test_y = train_set[test_index], label[test_index]

        X_train = torch.tensor(train_x, dtype=torch.float32).unsqueeze(1)
        y_train = torch.tensor(train_y, dtype=torch.long)
        X_test = torch.tensor(test_x, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(test_y, dtype=torch.long)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMNet(input_size, hidden_size, num_layers, num_classes).to(device)
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            epoch_end_time = time.time()
            if (epoch + 1) % 10 == 0:
                print(f'Fold {fold} Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Epoch Time: {epoch_end_time - epoch_start_time:.2f} seconds')

        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            accuracy_list.append(accuracy)
            fold_end_time = time.time()
            print(f'Fold {fold} Accuracy: {accuracy:.4f}, Fold Time: {fold_end_time - fold_start_time:.2f} seconds')

    kfold_end_time = time.time()
    avg_accuracy = np.mean(accuracy_list)
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f"K-Fold Cross Validation Time: {kfold_end_time - kfold_start_time:.2f} seconds")

    total_end_time = time.time()
    print(f"Total Run Time: {total_end_time - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
