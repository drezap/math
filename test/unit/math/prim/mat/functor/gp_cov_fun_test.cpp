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
    operator()(const Eigen::Matrix<T0__, -1, -1>& x,
               const T1__& sigma,
               const T2__& length_scale) const {
      typedef typename boost::math::tools::promote_args<T0__, T1__>::type scalar;
      std::vector<Eigen::Matrix<T0__, -1, 1>> x_in(x.cols());
      for (size_t i = 0; i < x.cols(); ++i) {
        x_in.resize(x.rows(), 1);
        x_in[i] = x.col(i);
      }
      Eigen::Matrix<scalar, -1, -1> cov =
        stan::math::gp_exp_quad_cov(x_in, sigma, length_scale);
      return cov;
    }
};

TEST(MathPrimFunctor, test_compute_cov) {
  Eigen::Matrix<double, -1, -1> x; x.resize(3, 3);
  x << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  
  stan::math::gp_lpdf <false, cov_exp_quad_functor,
                       double, double, stan::math::var>
    gp_lpdf(cov_exp_quad_functor());
  //  stan::math::gp_lpdf(L_cov_exp_quad_functor__());
  
}
