import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import os
import matplotlib.pyplot as plt

def interp3dim(raw_time, raw_data, new_time):
    '''
    Interp 3d data (acc, gyr, mag)
    :param raw_time:
    :type raw_time:
    :param raw_data:
    :type raw_data:
    :param new_time:
    :type new_time:
    :return:
    :rtype:
    '''
    return_data = np.zeros([new_time.shape[0], 3])
    if raw_time[0] > new_time[0] or raw_time[-1] < new_time[-1]:
        print(
            "illegal time:{0}~{1}, {2}~{3}".format(
                raw_time[0], new_time[0], raw_time[-1], new_time[-1]
            )
        )
        raise ValueError

    def remove_duplicates(arr1, arr2):
        _, idx = np.unique(arr1, return_index=True)
        arr1_unique = arr1[np.sort(idx)]
        arr2_unique = arr2[np.sort(idx)]
        return arr1_unique, arr2_unique

    for i in range(3):
        unique_time, unique_data = remove_duplicates(raw_time, raw_data[:, i])
        # f = interp1d(raw_time, raw_data[:, i], kind="cubic")
        f = interp1d(unique_time, unique_data, kind="cubic")
        return_data[:, i] = f(new_time)
    return return_data

def spectrum_analysis(signal, freq):
    # 对输入信号做FFT频谱分析，并输出归一化后的幅值
    L = len(signal)  # 数据长度
    T = 1.0 / freq  # 采样周期
    # 执行快速傅里叶变换 (FFT)
    fft_values = fft(signal, n=512)

    # 计算对应的频率
    frequencies = fftfreq(L, T)

    # 仅保留正频率部分
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    fft_values = np.abs(fft_values[positive_freq_idx])

    return frequencies, fft_values / np.max(fft_values)

def energy_ration_in_fft(freq_range, frequencies, fft_values):
    # 计算指定频率范围内的频谱能量占比
    low_freq = 0
    high_freq = freq_range

    # 获取对应的频率索引
    freq_indices = np.where((frequencies >= low_freq) & (frequencies <= high_freq))
    energy_spectrum = fft_values ** 2
    band_energy = np.sum(energy_spectrum[freq_indices])
    # 计算所有频率的总能量
    total_energy = np.sum(energy_spectrum)
    # 可视化归一化频谱能量分布（能量/总能量）
    energy_ratio = band_energy / total_energy

    return energy_ratio

def find_nearest(array, value):
    """找到 array 中最接近 value 的元素"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]