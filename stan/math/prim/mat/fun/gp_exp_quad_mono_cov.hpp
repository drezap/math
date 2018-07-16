#ifndef STAN_MATH_PRIM_MAT_FUN_GP_EXP_QUAD_MONO_COV_HPP
#define STAN_MATH_PRIM_MAT_FUN_GP_EXP_QUAD_MONO_COV_HPP

#include <math.h>
#include <stan/math/prim/mat/fun/distance.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/distance.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/inv_square.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

  template<typename T_x, typename T_m, typename T_l>
inline typename Eigen::Matrix<
    typename stan::return_type<T_x, T_sigma, T_l, T_p>::type, Eigen::Dynamic,
    Eigen::Dynamic>
  gp_exp_quad_mono_1d_cov(const std::vector<T_x> &x, // may be not template it it like this?
                       const T_m &magnitude, const T_l &length_scale) {
  
    using std::exp;
    // TODO check proper errors (domain/range)

    size_t x_size = x.size();
    size_t x_size2 = 2 * x_size;

    //this will need to be piped into backward compatable
    // Eigen::Matrix<typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
    //               Eigen::Dynamic, Eigen::Dynamic>
    //   exp_quad_cov = cov_exp_quad(x, magnitude, length_scale);
    
    Eigen::Matrix<typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
                Eigen::Dynamic, Eigen::Dynamic>
      cov(x_size2, x_size2);
    cov.block<x_size,x_size >(0, 0) = cov_exp_quad(x, magnitude, length_scale);
    
    T_m mag_sq = square(magnitude);
    
    
    for (int i = 0; i < x_size2; ++i) {
      for (int j = 0; j < x_size3; ++j) {
        
      }
    }
    
    

  return cov;
}

  // and we're going to assume non-ard here for simplicity
template<typename T_xd, typename T_x, typename T_m, typename T_l>
inline typename Eigen::Matrix<
    typename stan::return_type<T_x, T_xd, T_sigma, T_l, T_p>::type, Eigen::Dynamic,
    Eigen::Dynamic>
  gp_exp_quad_dx_f_cov(const std::vector<T_xd> &xd, const std::vector<T_x> &x,
                       const std::vector<T_l> &magnitude,
                       const T_l &length_scale){
  using std::exp;
  
  size_t x_size = xd.size();    
  size_t x_size = x.size();
  Eigen::Matrix<typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
                Eigen::Dynamic, Eigen::Dynamic>
    cov(xd_size, x_size);

  T_m mag_sq = square(magnitude);
  T_l neg_inv_l_sq = -inv_square(length_scale);
  for (int i = 0; i < xd_size; ++i) {
    for (int j = 0; j < x_size; ++j) {

      // may need to swap X and XD, i don't think this is
      // accurate
      cov(i, j) = mag_sq * exp(0.5 * inv_l_sq *
                               squared_distance(xd[i], xd[j])) *
        neg_inv_l_sq * squared_distance(x[i], x[j]);
    }
  }  
  return cov;
}


}  // namespace math
}  // namespace stan

#endif
