import numpy as np
import os
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import m2cgen as m2c
import joblib

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

def extract_features(acc_norm, gyro_norm, freq):
    acc_frequencies, acc_fft_values = spectrum_analysis(acc_norm, freq)
    gyro_frequencies, gyro_fft_values = spectrum_analysis(gyro_norm, freq)

    acc_ratio_5Hz = energy_ration_in_fft(5, acc_frequencies, acc_fft_values)
    acc_ratio_10Hz = energy_ration_in_fft(10, acc_frequencies, acc_fft_values)
    acc_ratio_20Hz = energy_ration_in_fft(20, acc_frequencies, acc_fft_values)
    acc_ratio_30Hz = energy_ration_in_fft(30, acc_frequencies, acc_fft_values)
    
    gyro_ratio_5Hz = energy_ration_in_fft(5, gyro_frequencies, gyro_fft_values)
    gyro_ratio_10Hz = energy_ration_in_fft(10, gyro_frequencies, gyro_fft_values)
    gyro_ratio_20Hz = energy_ration_in_fft(20, gyro_frequencies, gyro_fft_values)
    gyro_ratio_30Hz = energy_ration_in_fft(30, gyro_frequencies, gyro_fft_values)

    # 提取特征：均值和标准差
    # 提取特征：均值、标准差，以及各频率段的能量比率
    features = {
        'acc_mean': np.mean(acc_norm),
        'acc_std': np.std(acc_norm),
        'gyro_mean': np.mean(gyro_norm),
        'gyro_std': np.std(gyro_norm),
        'acc_ratio_5Hz': acc_ratio_5Hz,
        'acc_ratio_10Hz': acc_ratio_10Hz,
        'acc_ratio_20Hz': acc_ratio_20Hz,
        'acc_ratio_30Hz': acc_ratio_30Hz,
        'gyro_ratio_5Hz': gyro_ratio_5Hz,
        'gyro_ratio_10Hz': gyro_ratio_10Hz,
        'gyro_ratio_20Hz': gyro_ratio_20Hz,
        'gyro_ratio_30Hz': gyro_ratio_30Hz
    }

    return features

def find_nearest(array, value):
    """找到 array 中最接近 value 的元素"""
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def is_time_range_in_interval(time_range, intervals):
    """判断 time_range 是否完全在给定的 intervals 中"""
    start_time = time_range[0]
    end_time = time_range[-1]

    for (interval_start, interval_end, state) in intervals:
        if start_time >= interval_start and end_time <= interval_end:
            return state  # 返回对应的状态

    return None  # 如果不在任何区间内，返回 None

def interval_overlap(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    # 计算交集的起点和终点
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    # 如果交集是有效的区间，则返回长度，否则返回0
    return max(0, overlap_end - overlap_start)

def cal_vehicle_state_coverge(time_marker_onvehicle_intervals, state_manager_onvehicle_intervals):
    # 计算真实车辆状态的总长度
    real_vehicle_duration = 0
    for start, end, state in time_marker_onvehicle_intervals:
        if state == "OnVehicle":  # 只统计 OnVehicle 状态的时长
            real_vehicle_duration += end - start

    # 计算检测到的车辆状态与真实车辆状态的重叠部分
    overlap_duration = 0
    detected_vehicle_duration = 0  # 检测到的 OnVehicle 状态总时长

    for detected_interval in state_manager_onvehicle_intervals:
        # 统计检测到的 OnVehicle 总时长
        if detected_interval[2] == "OnVehicle":
            detected_vehicle_duration += detected_interval[1] - detected_interval[0]

    for real_interval in time_marker_onvehicle_intervals:
        for detected_interval in state_manager_onvehicle_intervals:
            # 如果都是 OnVehicle 状态，计算区间的重叠部分
            if real_interval[2] == "OnVehicle" and detected_interval[2] == "OnVehicle":
                overlap_duration += interval_overlap(real_interval[:2], detected_interval[:2])

    # 计算 OnVehicle 状态的准确率
    if detected_vehicle_duration > 0 or real_vehicle_duration > 0:
        accuracy = (overlap_duration / np.max([detected_vehicle_duration, real_vehicle_duration])) * 100
    else:
        accuracy = 0

    return accuracy

def cal_hand_state_coverge(time_marker_onhand_intervals, state_manager_onhand_intervals):
    # 计算真实 OnHand 状态的总长度
    real_hand_duration = 0
    for start, end, state in time_marker_onhand_intervals:
        if state == "OnHand":  # 只统计 OnHand 状态
            real_hand_duration += end - start

    # 计算检测到的 OnHand 状态与真实 OnHand 状态的重叠部分
    overlap_duration = 0
    detected_hand_duration = 0  # 检测到的 OnHand 状态总时长

    for detected_interval in state_manager_onhand_intervals:
        # 统计检测到的 OnHand 总时长
        if detected_interval[2] == "OnHand":
            detected_hand_duration += detected_interval[1] - detected_interval[0]

    for real_interval in time_marker_onhand_intervals:
        for detected_interval in state_manager_onhand_intervals:
            # 如果都是 OnHand 状态，计算区间的重叠部分
            if real_interval[2] == "OnHand" and detected_interval[2] == "OnHand":
                overlap_duration += interval_overlap(real_interval[:2], detected_interval[:2])

    # 计算 OnHand 状态的准确率
    if detected_hand_duration > 0 or real_hand_duration > 0:
        accuracy = (overlap_duration / np.max([detected_hand_duration, real_hand_duration])) * 100
    else:
        accuracy = 0

    return accuracy

def mtest_data(test_path, clf, hyperparameter):
    freq = hyperparameter['freq']
    fft_len = hyperparameter['fft_len']

    time_marker_path = os.path.join(test_path, "time_marker.txt")
    sensor_data_path = os.path.join(test_path, "sensors.txt")

    trans_time = np.loadtxt(time_marker_path, delimiter=',')
    trans_time = trans_time[:, 1]
    sensor_data = np.loadtxt(sensor_data_path, delimiter=',')

    gyro_data = sensor_data[sensor_data[:, 0] == 1]  # rad/s
    gyro_raw_time = gyro_data[:, 1]
    gyro_data = gyro_data[:, [2, 3, 4]]

    acc_data = sensor_data[sensor_data[:, 0] == 2]  # m/s2
    acc_raw_time = acc_data[:, 1]
    acc_data = acc_data[:, [2, 3, 4]]

    mag_data = sensor_data[sensor_data[:, 0] == 3]  # uT
    mag_raw_time = mag_data[:, 1]
    mag_data = mag_data[:, [2, 3, 4]]

    new_time_left = np.max([gyro_raw_time[0], acc_raw_time[0], mag_raw_time[0]])
    new_time_right = np.min([gyro_raw_time[-1], acc_raw_time[-1], mag_raw_time[-1]])
    new_time = np.linspace(new_time_left, new_time_right, int((new_time_right - new_time_left) * freq) + 1)

    gyro_data = interp3dim(gyro_raw_time, gyro_data, new_time)
    acc_data = interp3dim(acc_raw_time, acc_data, new_time)
    mag_data = interp3dim(mag_raw_time, mag_data, new_time)

    gyro_norm = np.linalg.norm(gyro_data, axis=1)
    acc_norm = np.linalg.norm(acc_data, axis=1)
    mag_norm = np.linalg.norm(mag_data, axis=1)

    time_marker_onvehicle_intervals = []  # 实际车辆状态区间
    time_marker_onhand_intervals = []  # 实际车辆状态区间
    # 将每对相邻的x_positions视为"车辆状态"的区间
    for i in range(0, len(trans_time) - 1, 2):
        start = find_nearest(new_time, trans_time[i])
        end = find_nearest(new_time, trans_time[i + 1])
        time_marker_onvehicle_intervals.append((start, end, "OnVehicle"))
    # 将每对相邻的x_positions视为"行人状态"的区间
    time_marker_onhand_intervals.append((new_time[0], trans_time[0], "OnHand"))
    for i in range(1, len(trans_time) - 1, 2):
        start = find_nearest(new_time, trans_time[i])
        end = find_nearest(new_time, trans_time[i + 1])
        time_marker_onhand_intervals.append((start, end, "OnHand"))
    time_marker_onhand_intervals.append((time_marker_onvehicle_intervals[-1][1], new_time[-1], "OnHand"))

    plt.figure()
    plt.plot(new_time, gyro_norm, label="Gyro Norm")

    # 在 time_marker.txt 中的时间点画竖线
    x_positions = np.loadtxt(os.path.join(test_path, "time_marker.txt"), delimiter=',')
    x_positions = x_positions[:, 1]
    for x in x_positions:
        plt.axvline(x=x, color='r', linestyle='--', linewidth=1, label="Time Marker")

    # 用于存储合并的 OnVehicle 区间
    on_vehicle_intervals = []

    # 用于标记当前的 OnVehicle 区间
    current_interval_start = None
    state_manager_intervals = []
    for i in range(0, len(gyro_norm) - fft_len, fft_len):
        feature = extract_features(acc_norm[i:(i + fft_len)], gyro_norm[i:(i + fft_len)], freq)
        state = clf.predict([list(feature.values())])[0]  # 获取预测状态
        start = new_time[i]
        end = new_time[i + fft_len]

        if state == "OnVehicle":
            # 如果当前区间是 OnVehicle 且还没有开始合并区间，则设置开始时间
            state_manager_intervals.append((start, end, "OnVehicle"))
            if current_interval_start is None:
                current_interval_start = start
        else:
            # 如果当前不是 OnVehicle 且有一个未结束的 OnVehicle 区间，则结束它
            if current_interval_start is not None:
                on_vehicle_intervals.append((current_interval_start, end))
                current_interval_start = None

        if state == "OnVehicle":
            # 如果当前区间是 OnVehicle 且还没有开始合并区间，则设置开始时间
            state_manager_intervals.append((start, end, "OnVehicle"))
        elif state == "OnHand":
            state_manager_intervals.append((start, end, "OnHand"))

    # 检查最后是否有未合并的 OnVehicle 区间
    if current_interval_start is not None:
        on_vehicle_intervals.append((current_interval_start, new_time[-1]))

    # 在图上绘制所有合并后的 OnVehicle 区间
    for interval_start, interval_end in on_vehicle_intervals:
        plt.axvspan(interval_start, interval_end, color='blue', alpha=0.3, label="StateManager OnVehicle")

    # 添加图例的处理，确保每个元素只添加一次
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 过滤重复的图例条目
    plt.legend(by_label.values(), by_label.keys())

    vehicle_state_coverage = cal_vehicle_state_coverge(time_marker_onvehicle_intervals, state_manager_intervals)
    hand_state_coverage = cal_hand_state_coverge(time_marker_onhand_intervals, state_manager_intervals)

    plt.title(f"OnVehicle {vehicle_state_coverage:.2f}%, OnHand {hand_state_coverage:.2f}%,")

    plt.show()


def data_prepare(root_path, hyperparameter):
    freq = hyperparameter['freq']
    fft_len = hyperparameter['fft_len']

    folders = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    features = []
    labels = []
    for _, folder_name in enumerate(folders):
        folder_path = os.path.join(root_path, folder_name)
        time_marker_path = os.path.join(folder_path, "time_marker.txt")
        sensor_data_path = os.path.join(folder_path, "sensors.txt")

        trans_time = np.loadtxt(time_marker_path, delimiter=',')
        trans_time = trans_time[:, 1]
        sensor_data = np.loadtxt(sensor_data_path, delimiter=',')

        gyro_data = sensor_data[sensor_data[:, 0] == 1]  # rad/s
        gyro_raw_time = gyro_data[:, 1]
        gyro_data = gyro_data[:, [2, 3, 4]]

        acc_data = sensor_data[sensor_data[:, 0] == 2]  # m/s2
        acc_raw_time = acc_data[:, 1]
        acc_data = acc_data[:, [2, 3, 4]]

        mag_data = sensor_data[sensor_data[:, 0] == 3]  # uT
        mag_raw_time = mag_data[:, 1]
        mag_data = mag_data[:, [2, 3, 4]]

        new_time_left = np.max([gyro_raw_time[0], acc_raw_time[0], mag_raw_time[0]])
        new_time_right = np.min([gyro_raw_time[-1], acc_raw_time[-1], mag_raw_time[-1]])
        new_time = np.linspace(new_time_left, new_time_right, int((new_time_right - new_time_left) * freq) + 1)

        gyro_data = interp3dim(gyro_raw_time, gyro_data, new_time)
        acc_data = interp3dim(acc_raw_time, acc_data, new_time)
        mag_data = interp3dim(mag_raw_time, mag_data, new_time)

        gyro_norm = np.linalg.norm(gyro_data, axis=1)
        acc_norm = np.linalg.norm(acc_data, axis=1)
        mag_norm = np.linalg.norm(mag_data, axis=1)

        time_marker_onvehicle_intervals = []  # 实际车辆状态区间
        time_marker_onhand_intervals = []  # 实际车辆状态区间
        # 将每对相邻的x_positions视为"车辆状态"的区间
        for i in range(0, len(trans_time) - 1, 2):
            start = find_nearest(new_time, trans_time[i])
            end = find_nearest(new_time, trans_time[i + 1])
            time_marker_onvehicle_intervals.append((start, end, "OnVehicle"))
        # 将每对相邻的x_positions视为"行人状态"的区间
        time_marker_onhand_intervals.append((new_time[0], trans_time[0], "OnHand"))
        for i in range(1, len(trans_time) - 1, 2):
            start = find_nearest(new_time, trans_time[i])
            end = find_nearest(new_time, trans_time[i + 1])
            time_marker_onhand_intervals.append((start, end, "OnHand"))
        time_marker_onhand_intervals.append((time_marker_onvehicle_intervals[-1][1], new_time[-1], "OnHand"))

        for i in range(fft_len, len(acc_data), fft_len):
            # 获取当前数据段的时间范围
            time_range = new_time[(i-fft_len):i]

            # 检查 time_range 是否在 OnHand 区间
            state = is_time_range_in_interval(time_range, time_marker_onhand_intervals)
            if state is None:
                # 如果不在 OnHand 区间，检查是否在 OnVehicle 区间
                state = is_time_range_in_interval(time_range, time_marker_onvehicle_intervals)

            # 如果状态不为 None，添加标签，否则跳过该数据段
            if state:
                labels.append(state)
                # 提取每段 IMU 数据的特征
                feature = extract_features(acc_norm[(i-fft_len):i], gyro_norm[(i-fft_len):i], freq)
                features.append(list(feature.values()))
            else:
                continue  # 跳过不在任何状态区间内的数据段

    # 将数据划分为训练集和测试集
    featurn_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

    return featurn_train, feature_test, label_train, label_test

def show_decision_tree(clf):
    # 生成特征名称
    feature_names = ['acc_mean',
        'acc_std',
        'gyro_mean',
        'gyro_std',
        'acc_ratio_5Hz',
        'acc_ratio_10Hz',
        'acc_ratio_20Hz',
        'acc_ratio_30Hz',
        'gyro_ratio_5Hz',
        'gyro_ratio_10Hz',
        'gyro_ratio_20Hz',
        'gyro_ratio_30Hz']
    plt.figure(figsize=(20, 10))  # 调整图像大小
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=['OnVehicle', 'OnHand'])
    plt.title("Decision Tree")
    plt.show()

if __name__ == "__main__":
    # 训练数据集
    root_path = "/home/sunxuehang/code/state_manager/data/决策树"
    # 超参数
    hyperparameter = {
        'freq':100.0,
        'fft_len':512
    }

    featurn_train, feature_test, label_train, label_test = data_prepare(root_path, hyperparameter)

    # 创建决策树分类器
    clf = DecisionTreeClassifier()
    # 训练模型
    clf.fit(featurn_train, label_train)
    # 进行预测
    y_pred = clf.predict(feature_test)
    # 评估模型准确率
    accuracy = accuracy_score(label_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # 将决策树模型转换为 C++ 代码
    cpp_code = m2c.export_to_c(clf)
    # 保存为 .cpp 文件
    with open("./decision_tree.cpp", "w") as f:
        f.write(cpp_code)
    # 保存模型到文件
    joblib.dump(clf, 'decision_tree_model.pkl')

    # 显示决策树结构
    # show_decision_tree(clf)

    # 测试决策树
    test_path = "/home/sunxuehang/code/state_manager/2024_09_22_12_13_01_157"
    clf_loaded = joblib.load('./decision_tree_model.pkl')
    mtest_data(test_path, clf_loaded, hyperparameter)