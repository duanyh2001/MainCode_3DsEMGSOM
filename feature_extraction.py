"""
File Name: feature_extraction.py
Description: Traditional features extraction for EMG signals
Author: dyh
Date: Apr 20, 2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftfreq
import pywt


def extract_data(raw_data, start_index=0):
    """Extract active(0)/rest(1) segment from raw emg data"""
    tensor_list = np.array([])
    for i in range(start_index, 21, 2):
        delete_tensor = np.arange(i*1000, (i+1)*1000)
        tensor_list = np.append(tensor_list, delete_tensor)

    tensor_list = np.stack(tensor_list).astype(np.int32)
    return np.delete(raw_data, tensor_list, axis=0)


def data_filter(x):
    '''
    系数= 2*f/Fs
    :param x: x的每行为一个时间序列

    :return: 滤波后的数据
    '''

    a, b = signal.butter(8, 0.9, 'highpass')
    # a, b = signal.butter(8, [0.45, 1.95], 'bandpass')
    x = signal.filtfilt(a, b, x)

    return x

def normalize_feature(feature_array):
    feature_array -= np.mean(feature_array)
    feature_array /= np.std(feature_array)
    return feature_array


class FeatureExtractor(object):
    def __init__(self, w_s=50, inc=10, data=None):
        self.window_size = w_s
        self.increment = inc  # sliding window increment
        self.epsilon = 10  # The threshold of delta value
        self.raw_data = data
        self._slide_window()

    def _slide_window(self):
        segment_data = []
        index = 0
        while index < len(self.raw_data):
            try:
                segment_data.append(self.raw_data[index:index + self.window_size])
                index = index + self.increment
            except:
                break

        self.segment_data = np.array(segment_data, dtype=object)

    def MOVING_AVG(self):
        emg_avg = np.sum(self.raw_data, axis=1)
        # print(np.shape(emg_avg))
        feature_list = np.zeros_like(emg_avg)
        for idx, item in enumerate(emg_avg):
            if idx >= 50:
                feature_list[idx] = np.mean(np.power(emg_avg[idx-50:idx], 2))
            else:
                feature_list[idx] = item**2

        return feature_list

    def MEAN(self):
        """Mean of a window data"""
        feature_list = []
        for data in self.segment_data:
            feature = np.mean(data, axis=0)
            feature_list.append(feature)

        return np.array(feature_list)

    def MAV(self):
        """Mean Absolute Value"""
        feature_list = []
        for data in self.segment_data:
            # print(np.shape(data))
            feature = np.mean(np.abs(data), axis=0)
            feature_list.append(feature)

        return np.array(feature_list)

    def RMS(self):
        """Root Mean Square"""
        feature_list = []
        for data in self.segment_data:
            # print(np.shape(data))
            feature = np.mean(np.square(data), axis=0)
            feature = np.sqrt(feature)
            feature_list.append(feature)

        return np.array(feature_list)

    def WL(self):
        """Waveform Length"""
        feature_list = []
        for data in self.segment_data:
            delta_value = []
            for i in range(1, len(data)):
                delta_value.append(np.abs(data[i] - data[i-1]))
            # print(np.shape(np.array(delta_value)))
            feature = np.sum(np.array(delta_value), axis=0)
            feature_list.append(feature)

        return np.array(feature_list, dtype=np.float64)

    def ZC(self):
        """Zero Crossing"""
        feature_list = []
        for data in self.segment_data:
            count = np.zeros([1, 8])
            for i in range(1, len(data)):
                delta_value = abs(data[i] - data[i-1])
                for j in range(len(delta_value)):
                    if delta_value[j] >= self.epsilon and (data[i][j]*data[i-1][j] < 0):
                        count[0][j] += 1
            feature_list.append(count[0])

        return np.array(feature_list)

    def SSC(self):
        """Slope Sign Changes"""
        feature_list = []
        for data in self.segment_data:
            count = np.zeros([1, 8])
            for i in range(1, len(data)-1):
                delta = (data[i]-data[i-1]) * (data[i]-data[i+1])
                for j in range(len(delta)):
                    if delta[j] >= self.epsilon:
                        count[0][j] += 1
            feature_list.append(count[0])

        return np.array(feature_list)

    # 利用小波分析进行特征分析
    def DWT(self):
        feature_list = []
        n = 0
        # 按照三个通道中取绝对值最大的，每个data中取得一个[ADDD ADDD]的1*8特征
        # ref = np.zeros(4)
        # ref1 = np.zeros(8)

        # 按照三个通道中取绝对值最大的，每个data中取得一个[ADDD； ADDD]的2 * 4特征
        # ref = np.zeros(4)
        # ref1 = np.zeros(4)

        # 取12维特征向量
        ref = np.zeros(8)
        # print(ref)
        # print(ref1)
        feature_maxabs = []
        for data in self.segment_data:
            feature = pywt.wavedec(data, 'bior3.7')
            # print(feature)
            # feature = pywt.wavedec(data, 'coif5', level=2)
            # print(feature)
            # print(len(feature), len(feature[0]), len(feature[0][1]))

            # 保存所有特征
            # feature_maxabs.append(feature)

            # 取12维特征向量
            i = 0 # 取feature的第i+1列
            for j in range(len(feature[i])):
                f_num = 0
                r_num = 0
                for k in range(len(feature[i][j])):
                    if abs(feature[i][j][k]) > abs(ref[i]):
                        f_num = f_num + 1
                    else:
                        r_num = r_num + 1
                if f_num > r_num:
                    ref = feature[i][j]
            feature_maxabs.append(ref)
            ref = np.zeros(8)

            # 按照三个通道中取绝对值最大的，每个data中取得一个[ADDD ADDD]的1*8特征，设前三个为三个通道
            # for i in range(len(feature)-1):  # len(feature)=level+1
            #     for j in range(len(feature[i])):  # len(feature[i]=50,40,30,20,10
            #         for k in range(0, len(feature[i][j]), 3):  # len(feature[i][j]=12
            #             if abs(feature[i][j][k]) > abs(feature[i][j][k+1]):
            #                 ref[int(k / 3)] = feature[i][j][k]
            #             else:
            #                 ref[int(k / 3)] = feature[i][j][k+1]
            #             if abs(ref[int(k / 3)]) < abs(feature[i][j][k+2]):
            #                 ref[int(k / 3)] = feature[i][j][k + 2]
            #         for m in range(len(ref)):
            #             if abs(ref[m]) > abs(ref1[i*4+m]):
            #                 ref1[i*4+m] = ref[m]
            # feature_maxabs.append(ref1)
            # ref1 = np.zeros(8)

            # 按照三个通道中取绝对值最大的，每个data中取得一个[ADDD； ADDD]的2 * 4特征 设前三个为三个通道
            # for i in range(len(feature)-1):  # len(feature)=level+1
            #     for j in range(len(feature[i])):  # len(feature[i]=50,40,30,20,10
            #         for k in range(0, len(feature[i][j]), 3):  # len(feature[i][j]=12
            #             if abs(feature[i][j][k]) > abs(feature[i][j][k+1]):
            #                 ref[int(k / 3)] = feature[i][j][k]
            #             else:
            #                 ref[int(k / 3)] = feature[i][j][k+1]
            #             if abs(ref[int(k / 3)]) < abs(feature[i][j][k+2]):
            #                 ref[int(k / 3)] = feature[i][j][k + 2]
            #         for m in range(len(ref)):
            #             if abs(ref[m]) > abs(ref1[m]):
            #                 ref1[m] = ref[m]
            # feature_maxabs.append(ref1)
            # ref1 = np.zeros(4)

            # 按照三个通道中取绝对值最大的，每个data中取得一个[ADDD； ADDD]的2 * 4特征设前四个为ADDD
            # for i in range(len(feature) - 1):  # len(feature)=level+1
            #     for j in range(len(feature[i])):  # len(feature[i]=50,40,30,20,10
            #         for k in range(0, int(len(feature[i][j])/3)):  # len(feature[i][j]=12
            #             if abs(feature[i][j][k]) > abs(feature[i][j][k + 4]):
            #                 ref[int(k)] = feature[i][j][k]
            #             else:
            #                 ref[int(k)] = feature[i][j][k + 4]
            #             if ref[int(k)] < abs(feature[i][j][k + 8]):
            #                 ref[int(k)] = feature[i][j][k + 8]
            #         for m in range(len(ref)):
            #             if ref[m] > ref1[m]:
            #                 ref1[m] = ref[m]
            # feature_maxabs.append(ref1)
            # ref1 = np.zeros(4)


        feature_list=feature_maxabs

        # print('feature_list', feature_list[0])
        # print(len(feature_list),len(feature_list[0]))#,len(feature_list[0][0]))

            # feature_list.append(feature)
            # print('feature shape', type(feature), type(feature[1]), type(feature[0][0]),type(feature[0][1]), len(feature[0][2]), len(feature[0][3]))
# 返回结果为level+1个数字，第一个数组为逼近系数数组，后面的依次是细节系数数组
        return np.array(feature_list)

    def DFT(self):
        # DFT变换离散傅里叶变换
        # L, N = 10, 100
        # L = self.segment_data.size
        N = 1800
        n = 0
        # x = self.segment_data
        feature_list = []
        # X=np.array()
        # X = np.array()
        # X = np.zeros((N,)) + np.zeros((N,)) * 1j  # 频域频谱
        # print('X type', type(X))
        # print('X shape', X.shape)
        X=[]
        # x1=0+0*1j
        x1=[]
        # print('self.segment_data size', self.segment_data.size)
        # print('self.segment_data shape', self.segment_data.shape)
        for k in range(N):
            for data in self.segment_data:  # 时域离散信号
                # print('data type', type(data))
                # print('data shape', data.shape)
                # print('data len', len(data))
                for i in range(len(data)):
                    # for j in range(3):
                    #     X[k] = X[k] + data * np.exp(-1j * n * k / N * 2 * np.pi)
                        x1 = x1 + data * np.exp(-1j * n * k / N * 2 * np.pi)
                        n = n + 1
            # n = 0
            X.append(x1)
        feature_list.append(X)
        return np.array(feature_list)



def delete_noise(data, thre=1000):
    delete_tensor = []
    for i in range(len(data)):
        if data[i] > thre:
            delete_tensor.append(i)

    return np.delete(data, delete_tensor)


if __name__ == '__main__':
    csv_name = 'D:/emg_data/dyh/trial_1/motion_1.csv'
    data = pd.read_csv(csv_name)
    # data = data_filter(data)
    # data = data0[:12000]
    emg_data = data.values
    emg_data = np.delete(emg_data, 0, axis=1)  # 把第一列的时间戳删除
    emg_data = extract_data(emg_data, start_index=1)
    feature_ex = FeatureExtractor(data=emg_data)
    moving_avg = feature_ex.MOVING_AVG()

    # fft_y = fft(moving_avg)
    # x = np.arange(len(moving_avg))
    # plt.plot(x, np.abs(fft_y))
    # plt.show()

    # moving_avg = data_filter(moving_avg.T).T
    moving_avg = delete_noise(moving_avg)
    print(np.mean(moving_avg))
    plt.plot(moving_avg)
    plt.show()

    # mav_feature = feature_ex.MAV()
    # rms_feature = feature_ex.RMS()
    # wl_feature = feature_ex.WL()
    # zc_feature = feature_ex.ZC()




