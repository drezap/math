#ifndef STAN_MATH_REV_MAT_FUN_GP_EXPONENTIAL_COV_HPP
#define STAN_MATH_REV_MAT_FUN_GP_EXPONENTIAL_COV_HPP

#include <cmath>
#include <stan/math/prim/scal/fun/exp.hpp>
#include <stan/math/prim/scal/fun/squared_distance.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <vector>

namespace stan {
namespace math {
template <typename T_x, typename T_s, typename T_l>
class gp_matern_3_2_cov_vari : public vari {
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
  gp_matern_3_2_cov_vari(const std::vector<T_x> &x, const T_s &sigma,
                         const T_l &length_scale)
      : vari(0.0), size_(x.size()), size_ltri_(size_ * (size_ - 1) / 2),
        l_d_(value_of(length_scale)), sigma_d_(value_of(sigma)),
        sigma_sq_d_(sigma_d_ * sigma_d_),
        dist_(ChainableStack::instance().memalloc_.alloc_array<double>(
            size_ltri_)),
        l_vari_(length_scale.vi_), sigma_vari_(sigma.vi_),
        cov_lower_(ChainableStack::instance().memalloc_.alloc_array<vari *>(
            size_ltri_)),
        cov_diag_(
            ChainableStack::instance().memalloc_.alloc_array<vari *>(size_)) {
    T_l root_3_inv_l = pow(3, 0.5) / l_d_;
  }
};

} // namespace math
} // namespace stan

#endif
