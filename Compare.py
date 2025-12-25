import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    TestInCppProject_state = "./TestInCppProject/status_log.txt"
    marker_path = os.path.join("./Data/TestData/2025_12_25_22_10_26_196", "time_marker.txt")

    cpp_state = np.loadtxt(TestInCppProject_state, delimiter=',')
    time_marker = np.loadtxt(marker_path, delimiter=',')[:, 1]

    plt.figure()
    plt.plot(cpp_state[:,0], cpp_state[:,1], marker='o', label='State Manager Predict')
    for i, t in enumerate(time_marker):
        plt.axvline(x=t, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f"time marker {i}")  # 竖线
    plt.xlabel("ts(s)")
    plt.ylabel("state")
    plt.legend()
    plt.grid()
    plt.show()

