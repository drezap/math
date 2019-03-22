#include <stan/math/prim/mat.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

struct L_cov_exp_quad_functor__ {
    template <typename T0__, typename T1__, typename T2__>
        Eigen::Matrix<typename boost::math::tools::promote_args<T0__, T1__, T2__>::type,
                      Eigen::Dynamic,Eigen::Dynamic>
    operator()(const std::vector<Eigen::Matrix<T0__, Eigen::Dynamic,1> >& x, // 1-D GP
               const T1__& sigma,
               const T2__& length_scale,
               std::ostream* pstream__) const {
      typedef typename boost::math::tools::promote_args<T0__, T1__>::type scalar;
      Eigen::Matrix<scalar, -1, -1> cov =
        gp_exp_quad_cov(x, sigma, length_scale);
      return cov;
    }
};

TEST(MathPrimFunctor, test_compute_cov) {
  // TODO: test insantiation
  // TODO: test calculation
  //  stan::math::gp_lpdf<true, F, double, double, stan::math::var> gp;
  //  stan::math::gp_lpdf <double>gp(1.0); // instantiation works

  //  std::function stan::math::gp_exp_quad_cov;
  stan::math::gp_lpdf <false,L_cov_exp_quad_functor__,
                       double, double, stan::math::var>gp();
  //  Eigen::Matrix<double, -1, -1> (*F)(Eigen::Matrix<double, -1,  -1>, double, double) =
  std::vector<double> x(3);
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;
  // stan::math::gp_lpdf<true,
  //                     std::function<Eigen::Matrix<double, -1, -1>
  //                                   (Eigen::Matrix<double, -1, -1>,
  //                                    double,
  //                                    double)>, double, double, double>
  
  
  //  gp.gp_compute_cov(x);
  
}

// struct simple_eq_functor_original {
//   template <typename T0, typename T1>
//   inline Eigen::Matrix<typename boost::math::tools::promote_args<T0, T1>::type,
//                        Eigen::Dynamic, 1>
//   operator()(const Eigen::Matrix<T0, Eigen::Dynamic, 1>& x,
//              const Eigen::Matrix<T1, Eigen::Dynamic, 1>& y,
//              const std::vector<double>& dat, const std::vector<int>& dat_int,
//              std::ostream* pstream__) const {
//     typedef typename boost::math::tools::promote_args<T0, T1>::type scalar;
//     Eigen::Matrix<scalar, Eigen::Dynamic, 1> z(2);
//     z(0) = x(0) - y(0) * y(1);
//     z(1) = x(1) - y(2);
//     return z;
//   }
// };
