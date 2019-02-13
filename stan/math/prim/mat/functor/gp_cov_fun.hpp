#ifndef STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP
#define STAN_MATH_PRIM_MAT_FUNCTOR_GP_COV_FUN_HPP


#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace math {

template <bool propto, typename F, typename T_y, typename T_X, typename T_theta>
struct gp_lpdf {

  typedef typename return_type<T_y, T_X, T_theta> T_return_type;
  typedef typename partials_return_type<T_y, T_X, T_theta> T_partials_return_type;

  const F& f_;
  const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y_;
  const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X_;
  const std::vector<T_theta> theta_;
  // TODO: possible instantiation: Eigen::Dynamic -> X_rows
  Eigen::Matrix<T_partials_return_type, Eigen::Dynamic, Eigen::Dynamic> C_dbl_; // covariance matrix

  Eigen::Matrix<T_partials_return_type, Eigen::Dynamic, 1> alpha_; // See R&W equation (5.9)
 
   // Constructor 
  gp_lpdf(const F& f, const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,  
                const Eigen::Matrix<T_X, Eigen::Dynamic, Eigen::Dynamic>& X,
               const std::vector<T_theta> theta_) {
    // NOTES: 
    // 
    
    // TODO: populate covariance matrix
    // TODO: get X_size
    size_t x_rows = X_.rows();
    
    for (int i = 0; i < x_rows; ++i) {
      for (int j = 0; j < x_rows; ++j) {  // symmetric covariance matrix
        C_dbl_(i, j) = f(X_(i, j), theta_); // TODO: add input for parameters
      }
    }
    // TODO: add f function that can read in parameters theta_
    //       or something that 

    // Dan's instructions
    // TODO: compute cholesky decomposition
    // TODO: populate derivative matrix for C
    // TODO: compute some products.. solves... blah
  }

  T_return_type operator()() {
    // Evaluate lpdf + gradient if necessary (using propto)
  }
};
  
}
}


#endif
