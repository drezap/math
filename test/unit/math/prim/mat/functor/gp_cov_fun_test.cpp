#include <stan/math/prim/mat.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

struct cov_exp_quad_functor {
    template <typename T0__, typename T1__, typename T2__>
        Eigen::Matrix<typename boost::math::tools::promote_args<T0__, T1__, T2__>::type,
                      Eigen::Dynamic,Eigen::Dynamic>
    operator()(const std::vector<T0__>& x,
               const T1__& sigma,
               const T2__& length_scale) const {
               //               ,std::ostream* pstream__) const {
      typedef typename boost::math::tools::promote_args<T0__, T1__>::type scalar;
      Eigen::Matrix<scalar, -1, -1> cov =
        stan::math::gp_exp_quad_cov(x, sigma, length_scale);
      return cov;
    }
};

TEST(MathPrimFunctor, test_compute_cov) {
  std::vector<double> x(3); // only use column vectors
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;
  stan::math::gp_lpdf <false, cov_exp_quad_functor,
                       double, double, stan::math::var>
    gp_lpdf(cov_exp_quad_functor(), x);
  //  stan::math::gp_lpdf(L_cov_exp_quad_functor__());
  
}
