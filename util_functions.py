"""
File Name: util_functions.py
Description: Util functions used in other python script

Author: Yahan DUAN (duanyh@mial.nankai.edu.cn; 2330126751@qq.com)
Date: 2023-11-27
Version: 1.0
Copyright: Copyright (c) 2024 Yahan DUAN

Revision History:
    2023-11-27 - Yahan DUAN - Initial version
"""
# import myo
import time
import socket
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from feature_extraction import FeatureExtractor
# from my_utils import DataSaver
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal



def extract_data(raw_data, start_index=0):
    """Extract active(0)/rest(1) segment from raw emg data"""
    tensor_list = np.array([])
    for i in range(start_index, 24, 2):
        delete_tensor = np.arange(i*1000, (i+1)*1000)
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = np.stack(tensor_list).astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


def feature_extract(input_data):
    feature_ex = FeatureExtractor(data=input_data)
    mav_feature = feature_ex.MAV()
    rms_feature = feature_ex.RMS()
    zc_feature = feature_ex.ZC()
    ssc_feature = feature_ex.SSC()

    return np.hstack([mav_feature, rms_feature, zc_feature, ssc_feature])


def tcp_client():
    # 定义客户端
    IP_ADDRESS = '190.168.1.128'  # 服务器IP地址
    PORT = 40005
    ADDR = (IP_ADDRESS, PORT)
    sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP socket
    sender.settimeout(10)
    try:
        sender.connect(ADDR)
    except socket.error:
        print("Socket error, reconnecting...")

    return sender


def mav_detect(emg_data, threhold=6):
    """Detect active segment according to MAV value of emg data"""
    vote_count = 1
    for i in range(0, 50, 10):
        mav_energy = np.mean(np.abs(emg_data[i:i+10]))
        if mav_energy > threhold:
            vote_count += 1

    if vote_count > 3:
        # print("This is an active segment!")
        return True
    else:
        print("In the REST state...")


def rms_detect(emg_data, threshold=3.5):
    feature_ex = FeatureExtractor(inc=1, data=emg_data)
    rms = feature_ex.RMS()
    rms_arr = np.mean(rms, axis=1)
    rms_index = [i for i, rms in enumerate(rms_arr) if rms > threshold]
    emg_data = np.mean(emg_data, axis=1)

    return rms_index


def data_filter(x):
    '''
    系数= 2*f/Fs
    :param x: x的每行为一个时间序列

    :return: 滤波后的数据
    '''

    a, b = signal.butter(8, 0.45, 'lowpass')
    # a, b = signal.butter(8, [0.45, 1.95], 'bandpass')
    x = signal.filtfilt(a, b, x)

    return x


def read_csv(root_diretory='classfication/CSV/realtime_test/zhs/trial1/'):
    data_list = []
    for i, csv_name in enumerate(os.listdir(root_diretory)):
        file_dir = root_diretory + csv_name
        emg_df = pd.read_csv(file_dir)
        emg_data = emg_df.values
        emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
        emg_data = extract_data(emg_data)  # 删除休息状态行
        emg_data = (data_filter(emg_data.T)).T  # 低通滤波
        # emg_data = emg_data[rms_detect(emg_data)]  # 提取数据活跃的有效段

        feature_arr = feature_extract(emg_data)
        feature_arr = scale(feature_arr)  # 对每一种动作的每一个通道的特征做归一化
        # 转化成DataFrame
        feature_df = pd.DataFrame(feature_arr)
        feature_df['label'] = i
        data_list.append(feature_df)

    emg_df = pd.concat(data_list, ignore_index=True)
    emg_df = emg_df.reindex(np.random.permutation(emg_df.index))

    X = emg_df.values[:, :-1]
    Y = emg_df.values[:, -1]
    return X, Y, emg_df


def moving_avg(raw_data):
    emg_avg = np.sum(raw_data, axis=1)
    # print(np.shape(emg_avg))
    feature_list = np.zeros_like(emg_avg)
    for idx, item in enumerate(emg_avg):
        if idx >= 50:
            feature_list[idx] = np.mean(np.power(emg_avg[idx-50:idx], 2))
        else:
            feature_list[idx] = item**2

    return feature_list


def analyze_cc():
    """Compute the correlation coeffcient among features """
    train_set, label, emg_df = read_csv()

    feature_df = emg_df.iloc[:, :-1]  # 去掉label
    feature_arr = feature_df.values
    mean_feature = []
    for i in range(0, 32, 8):
        mean_feature.append(np.mean(feature_arr[:, i:i + 8], axis=1))

    mean_feature = np.array(mean_feature).T
    mean_df = pd.DataFrame(mean_feature, columns=['RMS', 'MAV', 'ZC', 'SSC'])

    print(mean_df.head())

    # plt.figure(figsize=(10, 10))
    plt.figure()
    sns.heatmap(mean_df.corr(), annot=True)
    plt.show()


def analyze_cc_2():
    """Compute the correlation coeffcient among features according to channels"""
    _, _, emg_df = read_csv()
    feature_df = emg_df.iloc[:, :-1]  # 去掉label

    channel_arr = np.zeros([4, 4])
    for start_index in range(8):
        channel_index = pd.Index([0 + start_index, 8 + start_index, 16 + start_index, 24 + start_index])
        channel_1 = feature_df[channel_index]
        channel_arr = channel_arr + channel_1.corr().values


    plt.figure(figsize=(10, 10))
    sns.heatmap(channel_arr/8, annot=True)


if __name__ == '__main__':

    # csv_name = "classfication/CSV/realtime_test/zhs/trial1/zhs_4.csv"
    # data = pd.read_csv(csv_name).values
    # emg_data = np.delete(data, 0, axis=1)
    #
    # _, emg_data = extract_data(emg_data)
    # rms_index = rms_detect(emg_data)
    # print(np.shape(emg_data))
    # print(len(rms_index))
    #
    # emg_data = emg_data[rms_index]
    # emg_data = data_filter(emg_data.T)
    # print(emg_data)
    # feature_arr = feature_extract(emg_data.T)


    analyze_cc()


