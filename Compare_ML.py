"""
File Name: Compare_ML.py
Description: This is a script for gesture classification using the compared method SVM and LDA algorithm,
             which includes the following functions:
             -Data processing
             -Training of gesture classification model
             -Result display
Author: Yahan DUAN (duanyh@mial.nankai.edu.cn; 2330126751@qq.com)
Date: 2024-09-13
Version: 1.0
Copyright: Copyright (c) 2024 Yahan DUAN

Revision History:
    2024-09-13 - Yahan DUAN - Initial version
"""
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, normalize, scale
from sklearn.model_selection import train_test_split, KFold

from feature_extraction import FeatureExtractor
from util_functions import rms_detect, data_filter
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import time


root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/dyh/offline_data/trial4/'
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


if __name__ == '__main__':

    root_dir = 'D:/MyDocuments/GraduationDesign/data/continuous/rzh/offline_data/trial1/'
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
    data = normalize(data, norm='l2')

    # 初始化SVM和LDA模型
    svm_model = SVC()
    lda_model = LinearDiscriminantAnalysis()

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 用于存储SVM和LDA结果
    svm_accuracies = []
    lda_accuracies = []
    svm_times = []
    lda_times = []

    # 计算SVM平均准确率和测试时长
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 训练并测试SVM
        svm_model.fit(X_train, y_train)

        start_time = time.time()
        y_pred = svm_model.predict(X_test)
        end_time = time.time()

        svm_accuracies.append(accuracy_score(y_test, y_pred))
        svm_times.append(end_time - start_time)

    # 计算LDA平均准确率和测试时长
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 训练并测试LDA
        lda_model.fit(X_train, y_train)

        start_time = time.time()
        y_pred = lda_model.predict(X_test)
        end_time = time.time()

        lda_accuracies.append(accuracy_score(y_test, y_pred))
        lda_times.append(end_time - start_time)

    # 计算SVM和LDA的平均准确率和测试时长
    svm_avg_accuracy = np.mean(svm_accuracies)
    svm_avg_time = np.mean(svm_times)
    lda_avg_accuracy = np.mean(lda_accuracies)
    lda_avg_time = np.mean(lda_times)

    print(f"SVM的平均准确率为: {svm_avg_accuracy * 100:.2f}%, 平均测试时长为: {svm_avg_time*1000:.2f} ms")
    print(f"LDA的平均准确率为: {lda_avg_accuracy * 100:.2f}%, 平均测试时长为: {lda_avg_time*1000:.2f} ms")
