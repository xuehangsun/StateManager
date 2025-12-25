import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import m2cgen as m2c
import joblib

from Tools import interp3dim, spectrum_analysis, energy_ration_in_fft, find_nearest
from GenerateSyncData import sync_data

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

def data_prepare(root_path, hyperparameter):
    freq = hyperparameter['freq']
    fft_len = hyperparameter['fft_len']

    # 文件夹名与标签的映射，支持常见写法
    label_alias = {
        "car": "OnVehicle",
        "pedestrian": "OnHand"
    }

    folders = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    features = []
    labels = []

    for folder_name in folders:
        folder_path = os.path.join(root_path, folder_name)
        # 根据文件夹名推断标签
        normalized_name = folder_name.lower()
        label = None
        for key, state_label in label_alias.items():
            if key in normalized_name:
                label = state_label
                break
        if label is None:
            print(f"跳过未识别标签的文件夹：{folder_name}")
            continue

        # 在文件夹下查找所有 sensors.txt
        sensor_files = []
        for current_root, _, files in os.walk(folder_path):
            if "sensors.txt" in files:
                sensor_files.append(os.path.join(current_root, "sensors.txt"))

        for sensor_data_path in sensor_files:
            sensor_data = np.loadtxt(sensor_data_path, delimiter=',')

            gyro_data = sensor_data[sensor_data[:, 0] == 1]  # rad/s
            acc_data = sensor_data[sensor_data[:, 0] == 2]  # m/s2
            mag_data = sensor_data[sensor_data[:, 0] == 3]  # uT

            gyro_raw_time = gyro_data[:, 1]
            gyro_data = gyro_data[:, [2, 3, 4]]

            acc_raw_time = acc_data[:, 1]
            acc_data = acc_data[:, [2, 3, 4]]

            mag_raw_time = mag_data[:, 1]
            # mag_data = mag_data[:, [2, 3, 4]]

            new_time_left = np.max([gyro_raw_time[0], acc_raw_time[0], mag_raw_time[0]])
            new_time_right = np.min([gyro_raw_time[-1], acc_raw_time[-1], mag_raw_time[-1]])
            if new_time_right <= new_time_left:
                print(f"{sensor_data_path} 时间戳异常，已跳过")
                continue

            new_time = np.linspace(new_time_left, new_time_right, int((new_time_right - new_time_left) * freq) + 1)

            gyro_data = interp3dim(gyro_raw_time, gyro_data, new_time)
            acc_data = interp3dim(acc_raw_time, acc_data, new_time)
            # mag_data = interp3dim(mag_raw_time, mag_data, new_time)

            gyro_norm = np.linalg.norm(gyro_data, axis=1)
            acc_norm = np.linalg.norm(acc_data, axis=1)
            # mag_norm = np.linalg.norm(mag_data, axis=1)

            for i in range(fft_len, len(acc_data), fft_len):
                feature = extract_features(acc_norm[(i-fft_len):i], gyro_norm[(i-fft_len):i], freq)
                features.append(list(feature.values()))
                labels.append(label)

    # 将数据划分为训练集和测试集
    featurn_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

    return featurn_train, feature_test, label_train, label_test


def TestDTModel(test_path, clf, hyperparameter):
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
    # mag_data = interp3dim(mag_raw_time, mag_data, new_time)

    gyro_norm = np.linalg.norm(gyro_data, axis=1)
    acc_norm = np.linalg.norm(acc_data, axis=1)
    # mag_norm = np.linalg.norm(mag_data, axis=1)

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

    plt.show()


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
    root_path = os.path.join(os.getcwd(), "Data")
    test_path = "./Data/TestData/2025_12_25_22_10_26_196"
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

    # # 显示决策树结构
    # show_decision_tree(clf)

    # 测试决策树
    sync_data(test_path, hyperparameter)
    if test_path and os.path.exists(test_path):
        clf_loaded = joblib.load('./decision_tree_model.pkl')
        TestDTModel(test_path, clf_loaded, hyperparameter)
