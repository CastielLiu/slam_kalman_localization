/**
 * cubature_kalman_filter.hpp
 * @author wendao
 * 20/12/6
 **/
#ifndef KKL_CUBATURE_KALMAN_FILTER_X_HPP
#define KKL_CUBATURE_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>
#include <kalman/kalman_filter.hpp>

/**
 * @brief Cubature Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class CubatureKalmanFilterX : public KalmanFilter<T, System>
{
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;    //列向量
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
  using KalmanFilter<T, System>::state_dim;
  using KalmanFilter<T, System>::N;
  using KalmanFilter<T, System>::input_dim;
  using KalmanFilter<T, System>::measurement_dim;
  using KalmanFilter<T, System>::M;
  using KalmanFilter<T, System>::mean;
  using KalmanFilter<T, System>::cov;
  using KalmanFilter<T, System>::system;
  using KalmanFilter<T, System>::process_noise;
  using KalmanFilter<T, System>::measurement_noise;
  using KalmanFilter<T, System>::kalman_gain;

public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param cov                  initial covariance
   */
  CubatureKalmanFilterX(const System& _system, int _state_dim, int _input_dim, int _measurement_dim, 
                 const MatrixXt& _process_noise, const MatrixXt& _measurement_noise, 
                 const VectorXt& _mean, const MatrixXt& _cov):
    KalmanFilter<T, System>(_system, _state_dim, _input_dim, _measurement_dim, _process_noise, _measurement_noise, _mean, _cov),
    S(2 * _state_dim) //容积点个数
  {
    weights.resize(S, 1);
    cubature_points.resize(S, N);
    ext_weights.resize(S, 1);
    ext_cubature_points.resize(S, N + M);
    expected_measurements.resize(S, M);

    
    for (int i = 0; i < S; i++)
    {
      weights[i] = 1.0 / S;     // initialize weights for cubature filter
      ext_weights[i] = 1.0 / S; // weights for extended state space which includes error variances
    }      
      
  }

  /**
   * @brief predict  预测函数
   * @param control  input vector
   */
  virtual void predict(const VectorXt& control) override
  {
    // calculate cubature points
    this->ensurePositiveFinite(cov);
    computeCubaturePoints(mean, cov, cubature_points); //根据上一时刻的均值和方差计算cubature点

    for (int i = 0; i < S; i++) 
      cubature_points.row(i) = system.f(cubature_points.row(i), control); //根据系统方程传播cubature点

    const auto& Q = process_noise; //系统噪声|过程噪声

    // unscented transform
    VectorXt mean_pred(N);
    MatrixXt cov_pred(N, N);

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) 
      mean_pred += weights[i] * cubature_points.row(i).transpose();   //传播后的cubature点集均值

    for (int i = 0; i < S; i++) 
      cov_pred += weights[i] * cubature_points.row(i).transpose() * cubature_points.row(i);

    cov_pred -= mean_pred * mean_pred.transpose(); //传播后的cubature点集方差
    cov_pred += Q;                                      //加上过程噪声

    //得到预测值和预测协方差
    mean = mean_pred;
    cov = cov_pred;
    /*
    std::cout << "predict mean \r\n";
    std::cout << std::fixed << std::setprecision(2) << mean << std::endl;

    std::cout << "predict cov \r\n";
    std::cout << std::fixed << std::setprecision(2) << cov << std::endl;
*/
  }

  /**
   * @brief correct      校正函数
   * @param measurement  观测值
   */
  virtual void correct_try(const VectorXt& measurement)   //error
  {
    // create extended state space which includes error variances
    VectorXt ext_mean_pred = VectorXt::Zero(N + M, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + M, N + M);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(M, M) = measurement_noise;

    this->ensurePositiveFinite(ext_cov_pred); //模板基类成员函数，需使用this访问，或者 BaseClass<T, System>::ensurePositiveFinite
    computeCubaturePoints(ext_mean_pred, ext_cov_pred, ext_cubature_points); //根据预测均值和协方差以及测量噪声计算cubature点                                                                      //此时测量误差并未添加到cubature主体,而是存放于拓展部分

    // cubature transform
    expected_measurements.setZero();
    for (int i = 0; i < S; i++) {
      expected_measurements.row(i) = system.h(ext_cubature_points.row(i).transpose().topLeftCorner(N, 1));     //观测方程传播cubature点集
      expected_measurements.row(i) += ext_cubature_points.row(i).bottomRightCorner(M, 1);//添加测量噪声
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(M);
    for (int i = 0; i < S; i++) 
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i).transpose();  //传播后的cubature点集均值
   
    MatrixXt expected_measurement_cov = MatrixXt::Zero(M, M);
    for (int i = 0; i < S; i++)
      expected_measurement_cov += ext_weights[i] * expected_measurements.row(i).transpose() * expected_measurements.row(i);        

    expected_measurement_cov -= expected_measurement_mean * expected_measurement_mean.transpose(); //传播后的cubature点集方差
    expected_measurement_cov += measurement_noise;  //R = measurement_noise

    // calculated transformed covariance
    MatrixXt cross_cov = MatrixXt::Zero(N+M, M); //状态和观测的互协方差阵

    for(int i=0; i<S; ++i)
      cross_cov += ext_weights[i] * ext_cubature_points.row(i).transpose() * expected_measurements.row(i);
    cross_cov -= ext_mean_pred * expected_measurement_mean.transpose();

    kalman_gain = cross_cov * expected_measurement_cov.inverse(); //卡尔曼增益 = 互协方差/观测预测协方差

    //std::cout << "kalman_gain \r\n";
    //std::cout << std::fixed << std::setprecision(2) << kalman_gain << std::endl;

    VectorXt ext_mean = ext_mean_pred + kalman_gain * (measurement - expected_measurement_mean);          //最优估计
    MatrixXt ext_cov = ext_cov_pred - kalman_gain * expected_measurement_cov * kalman_gain.transpose();   //最优估计的协方差

    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  virtual void correct(const VectorXt& measurement) override
  {
    //预测得到的均值和协方差
    VectorXt mean_pred = VectorXt(mean);
    MatrixXt cov_pred  = MatrixXt(cov);

    this->ensurePositiveFinite(cov); //模板基类成员函数，需使用this访问，或者 BaseClass<T, System>::ensurePositiveFinite
    computeCubaturePoints(mean, cov, cubature_points); //根据预测均值和协方差以及测量噪声计算cubature点                                                                      //此时测量误差并未添加到cubature主体,而是存放于拓展部分

    // cubature transform
    expected_measurements.setZero();
    for (int i = 0; i < S; i++)
      expected_measurements.row(i) = system.h(cubature_points.row(i).transpose());     //观测方程传播cubature点集

    VectorXt expected_measurement_mean = VectorXt::Zero(M);
    for (int i = 0; i < S; i++) 
      expected_measurement_mean += weights[i] * expected_measurements.row(i).transpose();  //传播后的cubature点集均值
   
    MatrixXt expected_measurement_cov = MatrixXt::Zero(M, M);
    for (int i = 0; i < S; i++)
      expected_measurement_cov += weights[i] * expected_measurements.row(i).transpose() * expected_measurements.row(i);        

    expected_measurement_cov -= expected_measurement_mean * expected_measurement_mean.transpose(); //传播后的cubature点集方差
    expected_measurement_cov += measurement_noise;  //R = measurement_noise

    // calculated transformed covariance
    MatrixXt cross_cov = MatrixXt::Zero(N, M); //状态和观测的互协方差阵

    for(int i=0; i<S; ++i)
      cross_cov += weights[i] * cubature_points.row(i).transpose() * expected_measurements.row(i);
    cross_cov -= mean_pred * expected_measurement_mean.transpose();

    kalman_gain = cross_cov * expected_measurement_cov.inverse(); //卡尔曼增益 = 互协方差/观测预测协方差

    //std::cout << "kalman_gain \r\n";
    //std::cout << std::fixed << std::setprecision(2) << kalman_gain << std::endl;

    mean = mean_pred + kalman_gain * (measurement - expected_measurement_mean);           //最优估计
    cov  = cov_pred  - kalman_gain * expected_measurement_cov * kalman_gain.transpose();  //最优估计的协方差 
  }

  const MatrixXt& getSamplePoints() const { return cubature_points; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int S;  //容积点个数

public:
  VectorXt weights;          //容积点权重,2N个容积点，权重相同
  MatrixXt cubature_points;  //容积点

  VectorXt ext_weights;
  MatrixXt ext_cubature_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute cubature points
   * @param mean          mean
   * @param cov           covariance
   * @param cubature_points  calculated cubature points
   */
  void computeCubaturePoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& cubature_points) {
    const int n = mean.size(); //状态维度
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt(cov);
    MatrixXt P_chol = llt.matrixU();
    MatrixXt U = P_chol * sqrt(n);

    for (int i = 0; i < n; i++) 
    {
      cubature_points.row(  i) = mean + U.col(i);
      cubature_points.row(n+i) = mean - U.col(i);
    }
  }
};


#endif
