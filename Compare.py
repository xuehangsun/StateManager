import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Debounced_state = "./TestInCppProject/status_log.txt"
    Non_Debounced_state = "./TestInCppProject/status_log_raw.txt"
    marker_path = os.path.join("./Data/TestData/2025_12_25_22_10_26_196", "time_marker.txt")

    Debounced_state = np.loadtxt(Debounced_state, delimiter=',')
    Non_Debounced_state = np.loadtxt(Non_Debounced_state, delimiter=',')
    time_marker = np.loadtxt(marker_path, delimiter=',')[:, 1]

    plt.figure()
    plt.plot(Debounced_state[:,0], Debounced_state[:,1], marker='o', label='Debounced_state')
    plt.plot(Non_Debounced_state[:,0], Non_Debounced_state[:,1], marker='x', label='Non_Debounced_state')
    for i, t in enumerate(time_marker):
        plt.axvline(x=t, color='r', linestyle='--', linewidth=2, alpha=0.7)  # 竖线
    plt.axvline(x=time_marker[0], color='r', linestyle='--', linewidth=2, alpha=0.7, label=f"time marker")  # 竖线
    plt.xlabel("ts(s)")
    plt.ylabel("state")
    plt.legend()
    plt.grid()
    plt.show()

