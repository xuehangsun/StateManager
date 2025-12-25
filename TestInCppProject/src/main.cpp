#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <ctime>
#include <iomanip> 

#include <Eigen/Dense>
#include "so3.hpp"
#include "StateManager.h"

using namespace std;

int main(int argc, char **argv){
    
    cout << "data input ... "<< endl;

    // 输入已经同步的时间戳和传感器数据文件路径
    string input_file_path = "/home/sunxuehang/code/StateManager/StateManager/Data/TestData/2025_12_25_22_10_26_196/time-acc-gyr-mag-qwxyz.txt";
    
    // StateManager mStateMgn("string");
    StateManager mStateMgn;

    // 创建一个 ifstream 对象，用于打开文件
    std::ifstream infile(input_file_path);
    // 检查文件是否成功打开
    if (!infile) {
        std::cerr << "无法打开文件!" << std::endl;
        return 1; // 退出程序
    }
    std::string line; // 用于存储每一行的内容
    // 逐行读取文件，直到文件末尾
    size_t idx = 0;
    clock_t start_time = clock();
    while (std::getline(infile, line)) {
        // this_thread::sleep_for(chrono::milliseconds(10));
        
        // 输出读取的每一行数据
        // std::cout << line << std::endl;
        std::vector<double> values;
        std::stringstream ss(line);
        std::string token;
        // 使用getline按逗号分割字符串并转换为double存入vector
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stod(token)); // 转换为double后存入vector
        }
        double ts = values[0];
        idx++;
        Eigen::Vector3d acc(values[1], values[2], values[3]);
        Eigen::Vector3d gyr(values[4], values[5], values[6]);
        Eigen::Vector3d mag(values[7], values[8], values[9]);
        // qwxyz device to world
        Eigen::Quaterniond quat(values[10], values[11], values[12], values[13]);
        quat.normalize();
        Sophus::SO3d Rni(quat);

        mStateMgn.insertMeasurement(ts, idx, acc, gyr, mag, Rni);

        // switch (mStateMgn.getPhoneState()) {
        //     case PhoneState::HandHold:
        //         // cout <<  "HandHold" << endl;
        //         outfile << ts << ",HandHold" << std::endl;
        //         break;
        //     case PhoneState::VehicleMounted:
        //         // cout <<  "VehicleMounted" << endl;
        //         outfile << ts << ",VehicleMounted" << std::endl;
        //         break;
        //     default:
        //         cout <<  "Unknown" << endl;
        //         break;
        // }

    }
    clock_t end_time = clock();
    double time_taken = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "程序运行时间: " << time_taken << " 秒" << std::endl;

    cout << "检测到车载" <<  mStateMgn.VehicleMountedCount << "次" << endl;
    cout << "检测到手持" <<  mStateMgn.HandHoldCount << "次" << endl;

    cout << "process end!"<< endl;
    return 0;
}

