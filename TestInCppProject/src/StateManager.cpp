#include "StateManager.h"
#include "dj_fft.h"
#include <iomanip> 
#include <cstdlib>  // 包含 exit 函数

// 构造函数，默认初始化
StateManager::StateManager() : fft_len(512), sample_freq(100.0), atomic_phone_state_(PhoneState::HandHold) {
    fft_len = 512;
    sample_freq = 100.0;
    update_time = 1.0; // 每间隔1s进行一次状态更新
    atomic_phone_state_ = PhoneState::Unknown;
    delay_transfer = true; // 是否开启延迟切换
    delay_time = 4.0; // 延迟3s切换

    pre_ts = 0; //初始化上一时刻的时间

    for (int k = 0; k < fft_len / 2; ++k) {  // 初始化fft变化对应的频率
        float frequency = k * sample_freq / fft_len;      // 计算频率
        fft_frequency.push_back(frequency);
    }
    for (int k = 0; k < int(delay_time / update_time); ++k) {  // 初始化fft变化对应的频率
        delay_state.push_back(PhoneState::Unknown);
    }

    log_out_bool = true;

    state_outfile.open("../status_log.txt");
    state_outfile << std::fixed << std::setprecision(3);
    state_raw_outfile.open("../status_log_raw.txt");
    state_raw_outfile << std::fixed << std::setprecision(3);
    // fft_input_file.open("../fft_input.txt"); 
    // fft_input_file << std::fixed << std::setprecision(3);
    // fft_output_file.open("../fft_output.txt"); 
    // fft_output_file << std::fixed << std::setprecision(3);
    // decision_tree_input_file.open("../decision_tree_input.txt");
    // decision_tree_input_file << std::fixed << std::setprecision(6);
    // acc_gyr_norm_file.open("../acc_gyr_norm.txt");
    // acc_gyr_norm_file << std::fixed << std::setprecision(6);
}

// 插入测量数据
bool StateManager::insertMeasurement(double ts, size_t idx, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, const Eigen::Vector3d& mag, const SO3d& ahrs_Rni) {
    // 将新的测量数据插入到deque中
    RawMeasurement new_meas = {ts, idx, acc, gyr, mag, ahrs_Rni};
    raw_meas_deque_.emplace_back(new_meas);
    raw_ts.emplace_back(ts);
    raw_gyr_norm.emplace_back(gyr.norm());
    raw_acc_norm.emplace_back(acc.norm());
    raw_mag_norm.emplace_back(mag.norm());

    if(raw_ts.size() > fft_len)
        raw_ts.pop_front();
    if(raw_meas_deque_.size() > fft_len)
        raw_meas_deque_.pop_front();
    if(raw_gyr_norm.size() > fft_len)
        raw_gyr_norm.pop_front();
    if(raw_acc_norm.size() > fft_len)
        raw_acc_norm.pop_front();
    if(raw_mag_norm.size() > fft_len)
        raw_mag_norm.pop_front();

    // 每间隔update_time更新一次状态
    current_ts = ts;
    if(pre_ts == 0)
        pre_ts = ts;
    if(current_ts  - pre_ts < update_time || raw_meas_deque_.size() < fft_len){
        return true;
    }else{
        pre_ts = current_ts;
        this->flashPhoneState();
    }
        
    return true;
}


PhoneState StateManager::getPhoneState(){
    if(raw_meas_deque_.size() < fft_len)
        return PhoneState::Unknown;

    return atomic_phone_state_.load();
}


void StateManager::flashPhoneState(){
    std::vector<double> feature_input = extract_feature();
    // if(log_out_bool){
    //     decision_tree_input_file << feature_input[0];
    //     for(int k = 1; k<feature_input.size();k++){
    //         decision_tree_input_file << "," << feature_input[k];
    //     }
    //     decision_tree_input_file << endl;
    // }


    double output[2];  // 输出2个类别的概率
    double input[12];
    for(int i=0; i<feature_input.size();i++)
        input[i] = feature_input[i];
    decision_tree(input, output); // 调用决策树函数

    // 判断分类结果
    PhoneState temp_state;
    if (output[0] < output[1]) 
        temp_state = PhoneState::VehicleMounted;
    else 
        temp_state = PhoneState::HandHold;

    // 记录未消抖的原始判定结果
    if(log_out_bool){
        if(temp_state == PhoneState::VehicleMounted)
            state_raw_outfile << current_ts << ",1" << endl;
        else if(temp_state == PhoneState::HandHold)
            state_raw_outfile << current_ts << ",0" << endl;
        else
            state_raw_outfile << current_ts << ",-1" << endl;
    }

    // 延迟切换
    if(!delay_transfer) // 如果不是延迟切换，直接改变当前状态
        atomic_phone_state_ = temp_state;
    else{
        delay_state.emplace_back(temp_state);
        delay_state.pop_front();
        PhoneState first_PhoneState = delay_state[0];
        for(int i=0; i<delay_state.size();i++){
            if(delay_state[i] != first_PhoneState)
                break;
            if(i == delay_state.size()-1){
                atomic_phone_state_ = first_PhoneState;
            }
        }
    }

    if(atomic_phone_state_ == PhoneState::VehicleMounted){
        if(log_out_bool){
            std::cout << std::fixed << std::setprecision(3); // 设置固定小数位数和精度
            std::cout << "当前时间：" << current_ts << "分类结果：类别 1 (OnVehicle)" << output[0] << std::endl;
            state_outfile << current_ts << ",1" << endl;
            VehicleMountedCount++;
        }
    }
    else if(atomic_phone_state_ == PhoneState::HandHold){
        if(log_out_bool){
            std::cout << std::fixed << std::setprecision(3); // 设置固定小数位数和精度
            std::cout << "当前时间：" << current_ts  << "分类结果：类别 2 (OnHand)" << output[1] << std::endl;
            state_outfile << current_ts << ",0" << endl;
            HandHoldCount++;
        }
    }
    else
        ;

}

// 计算给定频率范围内FFT的能量比率
double StateManager::energy_ratio_in_fft(double freq_range, std::vector<float> fft_values) {
    double low_freq = 0.0;
    double high_freq = freq_range;

    std::vector<double> energy_spectrum(fft_values.size());
    
    for (size_t i = 0; i < fft_values.size(); ++i) {
        energy_spectrum[i] = std::pow(fft_values[i], 2);  // 能量谱是 FFT 值的平方
    }

    double band_energy = 0.0;
    double total_energy = std::accumulate(energy_spectrum.begin(), energy_spectrum.end(), 0.0);

    // 累计频带能量
    for (size_t i = 0; i < fft_frequency.size(); ++i) {
        if (fft_frequency[i] >= low_freq && fft_frequency[i] <= high_freq) {
            band_energy += energy_spectrum[i];
        }
    }

    // 计算能量比率
    double energy_ratio = (total_energy > 0) ? (band_energy / total_energy) : 0.0;

    return energy_ratio;
}

void StateManager::cal_fft_value(){
    // 计算并保存acc的fft
    std::vector<std::complex<float>> acc_xi(fft_len);
    for (int i = 0; i < fft_len; ++i) {
        float time = i / sample_freq;
        acc_xi[i] = std::complex<float>(raw_acc_norm[i], 0.0f); // 正弦波，虚数部分为0
    }
    std::vector<std::complex<float>> acc_fft_result = dj::fft1d(acc_xi, dj::fft_dir::DIR_FWD);
    // std::vector<std::complex<float>> acc_fft_result = fft_1d(acc_xi, fft_dir::FFT_DIR_FWD);
    acc_fft_value.clear();
    double max_acc_fft_value = 1.0;
    for (int k = 0; k < fft_len / 2; ++k) {  // 只需计算前半部分的频率
        float magnitude = std::abs(acc_fft_result[k]);  // 计算幅值
        float frequency = k * sample_freq / fft_len;      // 计算频率
        acc_fft_value.push_back(magnitude);
        if(magnitude > max_acc_fft_value)
            max_acc_fft_value = magnitude;
    }

    // 计算并保存gyr的fft
    std::vector<std::complex<float>> gyr_xi(fft_len);
    for (int i = 0; i < fft_len; ++i) {
        float time = i / sample_freq;
        gyr_xi[i] = std::complex<float>(raw_gyr_norm[i], 0.0f); // 正弦波，虚数部分为0
    }
    std::vector<std::complex<float>> gyr_fft_result = dj::fft1d(gyr_xi, dj::fft_dir::DIR_FWD);
    // std::vector<std::complex<float>> fwdfft = fft_1d(xi, fft_dir::FFT_DIR_FWD);
    gyr_fft_value.clear();
    double max_gyr_fft_value = 1.0;
    for (int k = 0; k < fft_len / 2; ++k) {  // 只需计算前半部分的频率
        float magnitude = std::abs(gyr_fft_result[k]);  // 计算幅值
        float frequency = k * sample_freq / fft_len;      // 计算频率
        gyr_fft_value.push_back(magnitude);
        if(magnitude > max_gyr_fft_value)
            max_gyr_fft_value = magnitude;
    }

    for(int k = 0; k < acc_fft_value.size(); k++){
        acc_fft_value[k] = acc_fft_value[k] / max_acc_fft_value;
        gyr_fft_value[k] = gyr_fft_value[k] / max_gyr_fft_value;
    }
}

std::vector<double> StateManager::extract_feature(){
    std::vector<double> feature_input;

    double acc_mean = std::accumulate(raw_acc_norm.begin(), raw_acc_norm.end(), 0.0) / raw_acc_norm.size();
    double acc_std = 0.0;
    for (double value : raw_acc_norm) {
        acc_std += (value - acc_mean) * (value - acc_mean);
    }
    acc_std = std::sqrt(acc_std / raw_acc_norm.size());

    double gyr_mean = std::accumulate(raw_gyr_norm.begin(), raw_gyr_norm.end(), 0.0) / raw_gyr_norm.size();
    double gyr_std = 0.0;
    for (double value : raw_gyr_norm) {
        gyr_std += (value - gyr_mean) * (value - gyr_mean);
    }
    gyr_std = std::sqrt(gyr_std / raw_gyr_norm.size());

    // // 保存fft变换的输入信号
    // if(log_out_bool){
    //     acc_gyr_norm_file << raw_acc_norm[0] ;
    //     for(int k = 1;k < raw_acc_norm.size();k++){
    //         acc_gyr_norm_file << "," << raw_acc_norm[k] ;
    //     }
    //     acc_gyr_norm_file << endl;
    //     acc_gyr_norm_file << raw_gyr_norm[0] ;
    //     for(int k = 1;k < raw_gyr_norm.size();k++){
    //         acc_gyr_norm_file << "," << raw_gyr_norm[k] ;
    //     }
    //     acc_gyr_norm_file << endl;
    // }

    cal_fft_value();

    double acc_ratio_5Hz = energy_ratio_in_fft(5, acc_fft_value);
    double acc_ratio_10Hz = energy_ratio_in_fft(10, acc_fft_value);
    double acc_ratio_20Hz = energy_ratio_in_fft(20, acc_fft_value);
    double acc_ratio_30Hz = energy_ratio_in_fft(30, acc_fft_value);

    double gyr_ratio_5Hz = energy_ratio_in_fft(5, gyr_fft_value);
    double gyr_ratio_10Hz = energy_ratio_in_fft(10, gyr_fft_value);
    double gyr_ratio_20Hz = energy_ratio_in_fft(20, gyr_fft_value);
    double gyr_ratio_30Hz = energy_ratio_in_fft(30, gyr_fft_value);

    feature_input.push_back(acc_mean);
    feature_input.push_back(acc_std);
    feature_input.push_back(gyr_mean);
    feature_input.push_back(gyr_std);
    feature_input.push_back(acc_ratio_5Hz);
    feature_input.push_back(acc_ratio_10Hz);
    feature_input.push_back(acc_ratio_20Hz);
    feature_input.push_back(acc_ratio_30Hz);
    feature_input.push_back(gyr_ratio_5Hz);
    feature_input.push_back(gyr_ratio_10Hz);
    feature_input.push_back(gyr_ratio_20Hz);
    feature_input.push_back(gyr_ratio_30Hz);

    return feature_input;
}


void StateManager::decision_tree(double* input, double* output) {
    double var0[2];

    const double arr0[2] = {0.0, 1.0};
    const double arr1[2] = {1.0, 0.0};

    if (input[3] <= 0.1738891378045082) {
        if (input[7] <= 0.9999985992908478) {
            if (input[10] <= 0.9990804195404053) {
                if (input[7] <= 0.999984085559845) {
                    memcpy(var0, arr0, 2 * sizeof(double)); // 使用已定义的数组
                } else {
                    if (input[1] <= 0.2545327767729759) {
                        if (input[7] <= 0.9999844133853912) {
                            if (input[6] <= 0.9998613893985748) {
                                memcpy(var0, arr0, 2 * sizeof(double));
                            } else {
                                memcpy(var0, arr1, 2 * sizeof(double));
                            }
                        } else {
                            memcpy(var0, arr0, 2 * sizeof(double));
                        }
                    } else {
                        memcpy(var0, arr1, 2 * sizeof(double));
                    }
                }
            } else {
                if (input[10] <= 0.9991457164287567) {
                    memcpy(var0, arr1, 2 * sizeof(double));
                } else {
                    memcpy(var0, arr0, 2 * sizeof(double));
                }
            }
        } else {
            if (input[3] <= 0.006988369859755039) {
                memcpy(var0, arr0, 2 * sizeof(double));
            } else {
                memcpy(var0, arr1, 2 * sizeof(double));
            }
        }
    } else {
        if (input[9] <= 0.9375980496406555) {
            memcpy(var0, arr0, 2 * sizeof(double));
        } else {
            if (input[7] <= 0.9990335702896118) {
                if (input[7] <= 0.9988672435283661) {
                    if (input[8] <= 0.9386856257915497) {
                        memcpy(var0, arr0, 2 * sizeof(double));
                    } else {
                        memcpy(var0, arr1, 2 * sizeof(double));
                    }
                } else {
                    memcpy(var0, arr0, 2 * sizeof(double));
                }
            } else {
                if (input[2] <= 0.11164955049753189) {
                    if (input[3] <= 0.1917634755373001) {
                        memcpy(var0, arr1, 2 * sizeof(double));
                    } else {
                        memcpy(var0, arr0, 2 * sizeof(double));
                    }
                } else {
                    if (input[0] <= 9.765235424041748) {
                        if (input[11] <= 0.99786177277565) {
                            memcpy(var0, arr0, 2 * sizeof(double));
                        } else {
                            memcpy(var0, arr1, 2 * sizeof(double));
                        }
                    } else {
                        if (input[11] <= 0.9989190697669983) {
                            if (input[11] <= 0.9989029169082642) {
                                memcpy(var0, arr1, 2 * sizeof(double));
                            } else {
                                memcpy(var0, arr0, 2 * sizeof(double));
                            }
                        } else {
                            memcpy(var0, arr1, 2 * sizeof(double));
                        }
                    }
                }
            }
        }
    }
    memcpy(output, var0, 2 * sizeof(double));
}
