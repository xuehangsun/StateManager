#ifndef STATEMANAGER_H
#define STATEMANAGER_H

// #include "Tools.h"
#include <string>
#include <Eigen/Dense>
#include "so3.hpp"
#include <deque>
#include <atomic>
#include <numeric>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Sophus;

enum class PhoneState{
    HandHold,
    VehicleMounted,
    Unknown,
};

class StateManager{
    public:
        StateManager();
        StateManager(const std::string& config_file_path);

        bool insertMeasurement(double ts, size_t idx,
        const Eigen::Vector3d& acc,
        const Eigen::Vector3d& gyr,
        const Eigen::Vector3d& mag,
        const SO3d& ahrs_Rni);

        PhoneState getPhoneState();

        double sample_freq; // 信号采样频率
        int fft_len; //傅里叶变换点数
        double update_time; //每间隔1s更新一次状态

        bool delay_transfer;
        double delay_time;

        bool log_out_bool;

        std::ofstream state_outfile;
        std::ofstream fft_input_file;
        std::ofstream fft_output_file;
        std::ofstream decision_tree_input_file;
        std::ofstream acc_gyr_norm_file;

        int HandHoldCount = 0;
        int VehicleMountedCount = 0;

    private:
        struct RawMeasurement{
            double ts;
            size_t idx;
            Eigen::Vector3d acc, gyro, mag;
            SO3d ahrs_Rni;
        };
        std::deque<RawMeasurement> raw_meas_deque_;

        std::atomic<PhoneState> atomic_phone_state_;

        double current_ts;
        double pre_ts;

        std::deque<double> raw_ts;
        std::deque<double> raw_acc_norm;
        std::deque<double> raw_gyr_norm;
        std::deque<double> raw_mag_norm;

        std::vector<float> fft_frequency; // fft结果对应的频率
        std::vector<float> acc_fft_value; // fft value
        std::vector<float> gyr_fft_value;

        std::deque<PhoneState> delay_state;

        void cal_fft_value(); // 计算acc和gyro的fft
        double energy_ratio_in_fft(double freq_range, std::vector<float> fft_values); // 计算给定频率范围内FFT的能量比率
        void flashPhoneState(); // 更新状态
        std::vector<double> extract_feature(); // 提取特征
        void decision_tree(double * input, double * output); // 决策树 

};


#endif