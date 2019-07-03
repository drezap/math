#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP


#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <iostream>

namespace stan {
namespace math {

template <bool propto, typename F, typename T_y, typename T_X, typename T_theta>
class gp_lpdf {
public:
  const F& f_;
  const std::vector<T_X>& X_;
  Eigen::Matrix<typename partials_return_type<T_y, T_X, T_theta>::type,
                  Eigen::Dynamic, Eigen::Dynamic> C_dbl_;
  gp_lpdf(const F& f_,
          const std::vector<T_X>& X_):
    f_(f_), X_(X_)
  {
    C_dbl_ = f_(X_, 1.0, 1.0);
  };
};
  
}
}


#endif
