#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP


#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <stan/math/prim/mat/fun/inverse.hpp>
#include <iostream>
#include <type_traits>
#include <Eigen/Dense>

namespace stan {
namespace math {

template <bool propto, typename F, typename T_y, typename T_X, typename T_theta>
class gp_lpdf {
public:
  const F& f_;
  const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X_;
   const Eigen::Matrix<T_y, Eigen::Dynamic, 1> & y;
  //  const std::vector<T_theta>& theta;
  Eigen::Matrix<typename partials_return_type<T_y, T_X, T_theta>::type,
                  Eigen::Dynamic, Eigen::Dynamic> C_dbl_;
  gp_lpdf(const F& f_,
          const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X_,
          Eigen::Matrix<T_y, Eigen::Dynamic, 1> y):
    f_(f_),
    X_(X_),
    y(y)
    //    theta(theta)
  {
    C_dbl_ = f_(X_, 1.0, 1.0);  // make this intake an arbitrary number of parameters
    Eigen::Matrix<typename partials_return_type<T_y, T_X, T_theta>::type,
                  Eigen::Dynamic, Eigen::Dynamic>
    C_chol_ = cholesky_decompose(C_dbl_);  // Do we want to save the covariance matrix?

    //    alpha = K^(-1) y
    Eigen::Matrix<typename partials_return_type<T_y, T_X, T_theta>::type,
                  Eigen::Dynamic, Eigen::Dynamic>
    C_inv_ = inverse(C_dbl_);
    Eigen::Matrix<typename return_type<T_y, T_X, T_theta>::type, Eigen::Dynamic, Eigen::Dynamic>
      alpha = C_inv_ * y;
  };
};
  
}
}


#endif
