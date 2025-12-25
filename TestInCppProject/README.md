# StateManager

## 初始化
StateManager mStateMgn("cfg_file_name");// 从config.yaml文件中初始化<br>
StateManager mStateMgn; // 使用默认参数初始化

### 初始化参数说明(可以不赋值，有默认参数)
double sample_freq; // 信号的采样频率 <br>
int fft_len; //傅里叶变换点数 默认 512 <br>
double update_time; // 间隔多久更新一次状态 <br>
bool delay_transfer; // 是否开启消抖 <br>
double delay_time; // 延迟时间（消抖参数），当前状态保持多长时间在更新 <br>
bool log_out_bool; // 是否开启日志记录， 如果开启，日至输出到../目录下 <br>

## 接口说明
// 插入100Hz的信号
bool insertMeasurement(double ts, size_t idx,const Eigen::Vector3d& acc,const Eigen::Vector3d& gyr,const Eigen::Vector3d& mag,const SO3d& ahrs_Rni); <br>
// 获取当前手机的状态
PhoneState getPhoneState(); <br>

## 注意事项
StateManager调用insertMeasurement后，自动更新状态，更新频率根据update_time确定。<br>
傅里叶变换点数必须是2^n。<br>
当数据不足时，会输出Unknown状态。<br>
