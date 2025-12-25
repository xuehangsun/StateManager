import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

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

def interpQuaternion(raw_time, raw_data, new_time):
    """
    Interp quaternion (quaternion)

    :param raw_time:
    :type raw_time:
    :param raw_data: [ w, x, y, z]
    :type raw_data:
    :param new_time:
    :type new_time:
    :return: [w, x, y, z ]
    :rtype:
    """
    order_index = np.argsort(raw_time, kind='quicksort')
    # print(order_index)
    raw_data = raw_data[order_index, :]
    raw_time = raw_time[order_index]

    ski_index_list = list()
    for i in range(raw_time.shape[0] - 1):
        if raw_time[i + 1] <= raw_time[i]:
            print(i, raw_time.shape, raw_time[i], raw_time[i + 1])
            ski_index_list.append(i + 1)
    ski_index_list.append(0)
    ski_index_list.append(len(ski_index_list) - 1)

    # print('totally skip frame: ', len(ski_index_list))
    # print("before :", raw_time.shape)
    raw_time = np.delete(raw_time, ski_index_list, axis=0)
    raw_data = np.delete(raw_data, ski_index_list, axis=0)
    # print("after:", raw_time.shape)

    qua = Rotation.from_quat(raw_data[:, [1, 2, 3, 0]])

    r_slerp = Slerp(raw_time, qua)
    qua_inter = r_slerp(new_time).as_quat()[:, [3, 0, 1, 2]]
    return qua_inter

def sync_data(root_path, hyperparameter):
    sensor_data = np.loadtxt(os.path.join(root_path, "sensors.txt"), delimiter=',')
    output_file = os.path.join(root_path, "time-acc-gyr-mag-qwxyz.txt")

    freq = hyperparameter['freq']

    ts = sensor_data[:, 0]  # s

    gyro_data = sensor_data[sensor_data[:, 0] == 1]  # rad/s
    gyro_raw_time = gyro_data[:, 1]
    gyro_data = gyro_data[:, [2, 3, 4]]

    acc_data = sensor_data[sensor_data[:, 0] == 2]  # m/s2
    acc_raw_time = acc_data[:, 1]
    acc_data = acc_data[:, [2, 3, 4]]

    mag_data = sensor_data[sensor_data[:, 0] == 3]  # uT
    mag_raw_time = mag_data[:, 1]
    mag_data = mag_data[:, [2, 3, 4]]

    qwxyz_data = sensor_data[sensor_data[:, 0] == 5]  # qwxyz
    qwxyz_raw_time = [0, 9e10]
    if qwxyz_data.shape[0] > 0:  # 确保数组有足够的列
        qwxyz_raw_time = qwxyz_data[:, 1]
        qwxyz_data = qwxyz_data[:, [2, 3, 4, 5]]

    new_time_left = np.max([gyro_raw_time[0], acc_raw_time[0], mag_raw_time[0], qwxyz_raw_time[0]])
    new_time_right = np.min([gyro_raw_time[-1], acc_raw_time[-1], mag_raw_time[-1], qwxyz_raw_time[-1]])
    new_time = np.arange(new_time_left, new_time_right, 1 / freq)

    acc_data = interp3dim(acc_raw_time, acc_data, new_time)
    gyro_data = interp3dim(gyro_raw_time, gyro_data, new_time)
    mag_data = interp3dim(mag_raw_time, mag_data, new_time)
    if qwxyz_data.shape[0] > 0:  # 确保数组有足够的列
        qwxyz_data = interpQuaternion(qwxyz_raw_time, qwxyz_data, new_time)
    else:
        qwxyz_data = np.zeros((len(acc_data), 4))
        qwxyz_data[:, 0] = 1

    new_time = new_time.reshape(-1, 1)
    save_data = np.concatenate([new_time, acc_data, gyro_data, mag_data, qwxyz_data], axis=1)

    np.savetxt(output_file, save_data, delimiter=',',
               fmt='%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f')

    pass

if __name__ == '__main__':
    root_path = "./Data/TestData/2025_12_25_22_10_26_196"

    hyperparameter = {
        'freq':100.0,
        'fft_len':512
    }

    sync_data(root_path, hyperparameter)
