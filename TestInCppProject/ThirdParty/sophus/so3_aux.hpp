//
// Created by steve on 2021/9/28.
//

#ifndef FULLVISUALSLAM_THIRDPARTY_SOPHUS_SO3_AUX_HPP_
#define FULLVISUALSLAM_THIRDPARTY_SOPHUS_SO3_AUX_HPP_

#include "so3.hpp"
#include "Eigen/Dense"

template <typename T>
Eigen::Matrix<T, 3, 3> JrInvSO3(const Sophus::SO3<T> &R){
  Eigen::Matrix<T, 3, 1> log_vec = R.log();
  double half_phi = 0.5 * log_vec.norm();
  Eigen::Matrix<T, 3, 1> a = log_vec / (2. * half_phi);
  double cot = std::cos(half_phi) / std::sin(half_phi);
  if (half_phi < 1e-8) {// phi is really small.
    return Eigen::Matrix<T, 3, 3>::Identity();
  } else {
    Eigen::Matrix<T, 3, 3> Jr_inv =
        half_phi * cot * Eigen::Matrix<T, 3, 3>::Identity() +
        (1. - half_phi * cot) * (a * a.transpose()) +
        half_phi * Sophus::SO3<T>::hat(a);
    return std::move(Jr_inv);
  }
};

template< typename T>
Eigen::Matrix<T, 3, 3> JrSO3(const Sophus::SO3<T> &R){
  Eigen::Matrix<T, 3, 1> log_vec = R.log();
  double phi = log_vec.norm();
  Eigen::Matrix<T, 3, 1> phi_unit = log_vec / (phi);
  if(phi < 1e-8){
    return Eigen::Matrix<T, 3, 3>::Identity();
  }else{
    Eigen::Matrix<T, 3, 3> Jr =
    Eigen::Matrix<T, 3, 3> ::Identity() - (1. - cos(phi)) / std::pow(phi, 2.)  *Sophus::SO3<T>::hat(log_vec)
    + (phi - sin(phi)) / std::pow(phi, 3.) * Sophus::SO3<T>::hat(log_vec) * Sophus::SO3<T>::hat(log_vec);
    return std::move(Jr);
  }
}

//template <typename T>
//Eigen::Matrix<T, 3, 3> JlSO3(const Sophus::SO3<T> &R){
//  Eigen::Matrix<T, 3, 1> log_vec = R.log();
//  double half_phi = 0.5 * log_vec.norm();
//  Eigen::Matrix<T, 3, 1> a = log_vec / (2. * half_phi);
//  double cot = std::cos(half_phi) / std::sin(half_phi);
//  if (half_phi < 1e-5) {// phi is really small.
//    return Eigen::Matrix<T, 3, 3>::Identity();
//  } else {
//    Eigen::Matrix<T, 3, 3> Jr_inv =
//        half_phi * cot * Eigen::Matrix<T, 3, 3>::Identity() +
//        (1. - half_phi * cot) * (a * a.transpose()) +
//        half_phi * Sophus::SO3<T>::hat(a);
//    return std::move(Jr_inv);
//  }
//};

#endif // FULLVISUALSLAM_THIRDPARTY_SOPHUS_SO3_AUX_HPP_
