#ifndef STAN_MATH_REV_MAT_FUN_GP_EXPONENTIAL_COV_HPP
#define STAN_MATH_REV_MAT_FUN_GP_EXPONENTIAL_COV_HPP

#include <boost/math/tools/promotion.hpp>
#include <boost/type_traits.hpp>
#include <type_traits>
#include <stan/math/prim/scal/fun/exp.hpp>
#include <stan/math/prim/scal/fun/distance.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <iostream>
#include <vector>

namespace stan {
namespace math {

template <typename T_x, typename T_s, typename T_l>
class gp_exponential_cov_vari : public vari {
 public:
  const size_t size_;
  const size_t size_ltri_;
  const double l_d_;
  const double sigma_d_;
  const double sigma_sq_d_;
  double *dist_;
  vari *l_vari_;
  vari *sigma_vari_;
  vari **cov_lower_;
  vari **cov_diag_;
  gp_exponential_cov_vari(const std::vector<T_x> &x, const T_s &sigma,
                          const T_l &length_scale)
      : vari(0.0),
        size_(x.size()),
        size_ltri_(size_ * (size_ - 1) / 2),
        l_d_(value_of(length_scale)),
        sigma_d_(value_of(sigma)),
        sigma_sq_d_(sigma_d_ * sigma_d_),
        dist_(ChainableStack::instance().memalloc_.alloc_array<double>(
            size_ltri_)),
        l_vari_(length_scale.vi_),
        sigma_vari_(sigma.vi_),
        cov_lower_(ChainableStack::instance().memalloc_.alloc_array<vari *>(
            size_ltri_)),
        cov_diag_(
            ChainableStack::instance().memalloc_.alloc_array<vari *>(size_)) {
    double neg_inv_l = -1.0 / l_d_;
    size_t pos = 0;
    for (size_t j = 0; j < size_; ++j) {
      for (size_t i = j + 1; i < size_; ++i) {
        double dist = distance(x[i], x[j]).val();
        dist_[pos] = dist;
        cov_lower_[pos]
            = new vari(sigma_sq_d_ * std::exp(dist_[pos] * neg_inv_l), false);
        ++pos;
      }
    }
    for (size_t i = 0; i < size_; ++i)
      cov_diag_[i] = new vari(sigma_sq_d_, false);
    std::cout << "first function called\n";
  }
  virtual void chain() {
    double adjl = 0;
    double adjsigma = 0;
    for (size_t i = 0; i < size_ltri_; ++i) {
      vari *el_low = cov_lower_[i];
      double prod_add = el_low->adj_ * el_low->val_;
      adjl += prod_add * dist_[i];
      adjsigma += prod_add;
    }
    for (size_t i = 0; i < size_; ++i) {
      vari *el = cov_diag_[i];
      adjsigma += el->adj_ * el->val_;
    }
    l_vari_->adj_ += adjl / (l_d_ * l_d_);
    sigma_vari_->adj_ += adjsigma * 2 / sigma_d_;
  }
};
template <typename T_x>
inline typename Eigen::Matrix<var, -1, -1> gp_exponential_cov(
    const std::vector<T_x> &x, const var &sigma, const var &l) {
  const char *function = "gp_exponential_cov";
  check_positive_finite(function, "sigma", sigma);
  check_positive_finite(function, "l", l);
  size_t x_size = x.size();
  for (size_t i = 0; i < x_size; ++i)
    check_not_nan(function, "x", x[i]);

  Eigen::Matrix<var, -1, -1> cov(x_size, x_size);
  if (x_size == 0)
    return cov;

  gp_exponential_cov_vari<T_x, var, var> *baseVari
      = new gp_exponential_cov_vari<T_x, var, var>(x, sigma, l);
  size_t pos = 0;
  for (size_t j = 0; j < x_size - 1; ++j) {
    for (size_t i = (j + 1); i < x_size; ++i) {
      cov.coeffRef(i, j).vi_ = baseVari->cov_lower_[pos];
      cov.coeffRef(j, i).vi_ = cov.coeffRef(i, j).vi_;
      ++pos;
    }
    cov.coeffRef(j, j).vi_ = baseVari->cov_diag_[j];
  }
  cov.coeffRef(x_size - 1, x_size - 1).vi_ = baseVari->cov_diag_[x_size - 1];
  return cov;
}
  ///// ARD implementation
template <typename T_x, typename T_s, typename T_l>
class gp_exponential_cov_vari<std::vector<Eigen::Matrix<T_x, -1, 1>>,
                              T_s, std::vector<T_l> > :
    public vari {
 public:
  const size_t size_;
  const size_t size_ltri_;
  const size_t size_l_;
  const std::vector<double> l_d_;
  const double sigma_d_;
  const double sigma_sq_d_;
  double *dist_;
  vari *l_vari_;
  vari *sigma_vari_;
  vari **cov_lower_;
  vari **cov_diag_;
  gp_exponential_cov_vari(const std::vector<Eigen::Matrix<T_x, -1, 1>> &x,
                          const T_s &sigma,
                          const std::vector<T_l> &length_scale)
      : vari(0.0),
        size_(x.size()),
        size_ltri_(size_ * (size_ - 1) / 2),
        size_l_(length_scale.size()),
        l_d_(value_of(length_scale)),
        sigma_d_(value_of(sigma)),
        sigma_sq_d_(sigma_d_ * sigma_d_)// ,
        dist_(ChainableStack::instance().memalloc_.alloc_array<double>(
            size_ltri_)),
        l_vari_(ChainableStack::instance().memalloc_.alloc_array<vari*>(
            size_l_)),
        sigma_vari_(sigma.vi_),
        cov_lower_(ChainableStack::instance().memalloc_.alloc_array<vari *>(
            size_ltri_)),
        cov_diag_(
            ChainableStack::instance().memalloc_.alloc_array<vari *>(size_))
  {
    double neg_inv_l = -1.0 / l_d_;
    size_t pos = 0;
    for (size_t j = 0; j < size_; ++j) {
      for (size_t i = j + 1; i < size_; ++i) {
        auto dist = distance(x[i], x[j]).val();
        dist_[pos] = dist;
        cov_lower_[pos]
            = new vari(sigma_sq_d_ * std::exp(dist_[pos] * neg_inv_l), false);
        ++pos;
      }
    }
    for (size_t i = 0; i < size_; ++i)
      cov_diag_[i] = new vari(sigma_sq_d_, false);
    std::cout << "ard function called \n";
  }
  virtual void chain() {
    double adjl = 0;
    double adjsigma = 0;
    for (size_t i = 0; i < size_ltri_; ++i) {
      vari *el_low = cov_lower_[i];
      double prod_add = el_low->adj_ * el_low->val_;
      adjl += prod_add * dist_[i];
      adjsigma += prod_add;
    }
    for (size_t i = 0; i < size_; ++i) {
      vari *el = cov_diag_[i];
      adjsigma += el->adj_ * el->val_;
    }
    l_vari_->adj_ += adjl / (l_d_ * l_d_);
    sigma_vari_->adj_ += adjsigma * 2 / sigma_d_;
  }
};

template <typename T_x>
inline typename Eigen::Matrix<var, -1, -1>
gp_exponential_cov(const std::vector<Eigen::Matrix<T_x, -1, 1>> &x,
                   const var &sigma, const std::vector<var> &l) {
  const char *function = "gp_exponential_cov";
  check_positive_finite(function, "sigma", sigma);
  check_positive_finite(function, "l", l);
  size_t x_size = x.size();
  // for (size_t i = 0; i < x_size; ++i)
  //   check_not_nan(function, "x", x);

  Eigen::Matrix<var, -1, -1> cov(x_size, x_size);
  if (x_size == 0)
    return cov;

  gp_exponential_cov_vari<std::vector<Eigen::Matrix<T_x, -1, 1>>,
                          var, std::vector<var>>
    *baseVari =
    new gp_exponential_cov_vari<
      std::vector<Eigen::Matrix<T_x, -1, 1>>, var,
    std::vector<var>>(x, sigma, l);

  size_t pos = 0;
  for (size_t j = 0; j < x_size - 1; ++j) {
    for (size_t i = (j + 1); i < x_size; ++i) {
      cov.coeffRef(i, j).vi_ = baseVari->cov_lower_[pos];
      cov.coeffRef(j, i).vi_ = cov.coeffRef(i, j).vi_;
      ++pos;
    }
    cov.coeffRef(j, j).vi_ = baseVari->cov_diag_[j];
  }
  cov.coeffRef(x_size - 1, x_size - 1).vi_ = baseVari->cov_diag_[x_size - 1];
  return cov;
}

}  // namespace math
}  // namespace stan

#endif
