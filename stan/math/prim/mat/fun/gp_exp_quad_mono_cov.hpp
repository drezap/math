#ifndef STAN_MATH_PRIM_MAT_FUN_GP_EXP_QUAD_MONO_COV_HPP
#define STAN_MATH_PRIM_MAT_FUN_GP_EXP_QUAD_MONO_COV_HPP

#include <cmath>
#include <math.h>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/distance.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/distance.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/inv_square.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <vector>

namespace stan {
namespace math {

template <typename T_x, typename T_m, typename T_l>
inline typename Eigen::Matrix<
    typename stan::return_type<T_x, T_sigma, T_l, T_p>::type, Eigen::Dynamic,
    Eigen::Dynamic>
gp_exp_quad_mono_cov(const std::vector<T_x> &x, const T_m &magnitude,
                     const T_l &length_scale) {

  using std::exp;
  // TODO check proper errors (domain/range)

  size_t x_size = x.size();
  size_t x_size2 = 2 * x_size;

  // this will need to be piped into backward compatable
  Eigen::Matrix<typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
                Eigen::Dynamic, Eigen::Dynamic>
      exp_quad_cov = cov_exp_quad

          Eigen::Matrix<
              typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
              Eigen::Dynamic, Eigen::Dynamic>
              cov(x_size2, x_size2);

  T_m mag_sq = square(magnitude);

  for (int i = 0; i < x_size2; ++i) {
    for (int j = 0; j < x_size3; ++j) {
    }
  }

  return cov;
}

} // namespace math
} // namespace stan

#endif
