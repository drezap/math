#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP


#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>

namespace stan {
namespace math {

  template <bool propto, typename F, typename T_y, typename T_X, typename T_theta>
  class gp_lpdf {
public:
  //  typename stan::return_type<T_y, T_X, T_theta> T_return_type;
  //typedef typename partials_return_type<T_y, T_X, T_theta> T_partials_return_type;


  // TODO: all of these must be initialized in the constructor
  //  const F& f_;
  // const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y_;
  // const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X_;
  // const std::vector<T_theta> theta_;
  // // TODO: possible instantiation: Eigen::Dynamic -> X_rows
  // Eigen::Matrix<typename partials_return_type<T_y, T_X, T_theta>::type,
  //               Eigen::Dynamic, Eigen::Dynamic> C_dbl_; // covariance matrix

  // Eigen::Matrix<T_partials_return_type, Eigen::Dynamic, 1> alpha_; // See R&W equation (5.9)
 
  // Constructor 
  gp_lpdf()
          // ,const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,  
          //       const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X,
          // const std::vector<T_theta> theta_) :
  {
    
    // NOTES: 
    //
    // TODO: add f function that can read in parameters theta_
    //       or something that 
    // C_dpl_ = gp_compute_cov(f*
    // Dan's instructions
    // TODO: compute cholesky decomposition
    //       gp_compute_cov(*f, X_);
    // TODO: populate derivative matrix for C
    // TODO: compute some products.. solves... blah
  };

  
  // Eigen::Matrix<T_x, Eigen::Dynamic, 1> gp_compute_cov(const F& f,
  //                                                      const Eigen::Matrix<T_X,
  //                                                      Eigen::Dynamic,
  //                                                      Eigen::Dynamic>& X) {

  //   size_t x_size = X.rows();
  //   Eigen::Matrix<T_x, x_size, x_size> cov;
  //   for (size_t i = 0; i < x_size; ++i) {
  //     for (size_t j = 0; j < x_size; ++j) {
  //       cov = f(X(i, j));
  //     }
  //   }
  //   std::cout << cov << "\n";
  // }
  
  
  // T_return_type operator()() {
  //   // Evaluate lpdf + gradient if necessary (using propto)
  // }
};
  
}
}


#endif
