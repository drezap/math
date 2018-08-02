#ifndef STAN_MATH_REV_MAT_FUN_GP_DOT_PROD_COV_HPP
#define STAN_MATH_REV_MAT_FUN_GP_DOT_PROD_COV_HPP

#include <boost/math/tools/promotion.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/prim/scal/fun/exp.hpp>
#include <stan/math/prim/scal/fun/squared_distance.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <vector>

namespace stan {
namespace math {
/**
 * This is a subclass of the vari class for precomputed
 * gradients of gp_dot_prod_cov
 *
 * The class stores the double values for the distance
 * matrix, pointers to the varis for the covariance
 * matrix, along with a pointer to the vari for sigma,
 * and the vari for l.
 *
 * @tparam T_x type of std::vector of elements
 * @tparam T_sigma type of sigma
 * @tparam T_l type of length scale
 */
template <typename T_x, typename T_s>
class gp_dot_prod_cov_vari : public vari {
 public:
  const size_t size_;
  const size_t size_ltri_;
  const double sigma_d_;
  const double sigma_sq_d_;
  double *dist_;
  vari *sigma_vari_;
  vari **cov_lower_;
  vari **cov_diag_;
  /**
   * Constructor for gp_dot_prod_cov
   *
   * All memory allocated in
   * ChainableStack's stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled to the second argument to
   * vari's constructor.
   *
   * @param x std::vector input that can be used in square distance
   *    Assumes each element of x is the same size
   * @param sigma standard deviation
   * @param length_scale length scale
   */
  gp_dot_prod_cov_vari(const std::vector<T_x> &x, const T_s &sigma)
      : vari(0.0),
        size_(x.size()),
        size_ltri_(size_ * (size_ - 1) / 2 + size_),
        sigma_d_(value_of(sigma)),
        sigma_sq_d_(sigma_d_ * sigma_d_),
        sigma_vari_(sigma.vi_),
        cov_lower_(ChainableStack::instance().memalloc_.alloc_array<vari *>(size_ltri_)),
        cov_diag_(
            ChainableStack::instance().memalloc_.alloc_array<vari *>(size_)) {
    size_t pos = 0;
    for (size_t j = 0; j < size_; ++j) {
      for (size_t i = j; i < size_; ++i) {
        cov_lower_[pos] = new vari(sigma_sq_d_ + x[i].val() * x[j].val(),
                                   false);
        ++pos;
      }
    }
    // for (size_t i = 0; i < size_; ++i)
    //   cov_diag_[i] = new vari(sigma_sq_d_, false);
  }
  virtual void chain() {
    double adjsigma = 0;
    for (size_t i = 0; i < size_ltri_; ++i) {
      vari *el_low = cov_lower_[i];
      double prod_add = el_low->adj_ * el_low->val_;
      adjsigma += prod_add;
    }
    // for (size_t i = 0; i < size_; ++i) {
    //   vari *el = cov_diag_[i];
    //   adjsigma += el->adj_ * el->val_;
    // }
    sigma_vari_->adj_ += adjsigma * 2 / sigma_d_;
  }
};
/**
 * Returns an dot_prod kernel.
 *
 * @param x std::vector input that can be used in square distance
 *    Assumes each element of x is the same size
 * @param sigma standard deviation
 * @param l length scale
 * @return squared distance
 * @throw std::domain_error if sigma <= 0, l <= 0, or
 *   x is nan or infinite
 */
  template <typename T_x>
  inline Eigen::Matrix<stan::math::var, -1, -1>
gp_dot_prod_cov(const std::vector<T_x> &x, const var &sigma) {
  check_positive("gp_dot_prod_cov", "sigma", sigma);
  size_t x_size = x.size();
  for (size_t i = 0; i < x_size; ++i)
    check_not_nan("gp_dot_prod_cov", "x", x[i]);

  Eigen::Matrix<var, -1, -1> cov(x_size, x_size);
  if (x_size == 0)
    return cov;

  gp_dot_prod_cov_vari<T_x, var> *baseVari
      = new gp_dot_prod_cov_vari<T_x, var>(x, sigma);

  size_t pos = 0;
  for (size_t j = 0; j < x_size; ++j) {
    for (size_t i = (j + 1); i < x_size; ++i) {
      cov.coeffRef(i, j).vi_ = baseVari->cov_lower_[pos];
      cov.coeffRef(j, i).vi_ = cov.coeffRef(i, j).vi_;
      ++pos;
    }
    cov.coeffRef(j, j).vi_ = baseVari->cov_diag_[j];
  }
  return cov;
}
}  // namespace math
}  // namespace stan

#endif
