#ifndef STAN__VB__LATENT_VARS__HPP
#define STAN__VB__LATENT_VARS__HPP

#include <vector>

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/mdivide_left_tri_low.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/math/error_handling/matrix/check_cholesky_factor.hpp>

namespace stan {

  namespace vb {

    class latent_vars {

    private:

      Eigen::VectorXd mu_;    // Mean of location-scale family
      Eigen::MatrixXd L_;     // Lower-triangular decomposition of scale matrix
      Eigen::MatrixXd L_bar_; // Unique lower-triangular decomposition of scale matrix
      int dimension_;

    public:

      latent_vars(Eigen::VectorXd const& mu, Eigen::MatrixXd const& L) :
      mu_(mu), dimension_(mu.size()) {

        static const char* function = "stan::vb::latent_vars(%1%)";

        set_L_bar(L);

        double tmp(0.0);
        stan::math::check_cholesky_factor(function, L_, "Scale matrix", &tmp);
        stan::math::check_square(function, L_, "Scale matrix", &tmp);
        stan::math::check_size_match(function,
                                     L_.rows(),  "Dimension of scale matrix",
                                     dimension_, "Dimension of mean vector",
                                     &tmp);

      };

      virtual ~latent_vars() {}; // No-op

      // Accessors
      int dimension() const { return dimension_; }
      Eigen::VectorXd const& mu() const { return mu_; }
      Eigen::MatrixXd const& L() const { return L_; }
      Eigen::MatrixXd const& L_bar() const { return L_bar_; }

      // Mutators
      void set_mu(Eigen::VectorXd const& mu) { mu_ = mu; }
      void set_L_bar(Eigen::MatrixXd const& L_bar)
      {
        L_bar_ = L_bar;
        L_     = L_bar_;
        for (int d = 0; d < dimension_; ++d)
        {
          L_(d,d) = log(1+exp(L_bar_(d,d)));
        }
      }


      // Implements g^{-1}(\check{z}) = L\check{z} + \mu
      Eigen::VectorXd to_unconstrained(Eigen::VectorXd const& x) const {
        static const char* function = "stan::vb::latent_vars"
                                      "::to_unconstrained(%1%)";

        double tmp(0.0);
        stan::math::check_size_match(function,
                         x.size(), "Dimension of input vector",
                         dimension_, "Dimension of mean vector",
                         &tmp);

        Eigen::VectorXd result = L_*x + mu_;

        return result;
      };

      // // Implements g(\widetilde{z}) = L^{-1}(\check{z} - \mu)
      // void to_standardized(Eigen::VectorXd& x) const {
      //   static const char* function = "stan::vb::latent_vars"
      //                                 "::to_standardized(%1%)";

      //   double tmp(0.0);
      //   stan::math::check_size_match(function,
      //                    x.size(), "Dimension of input vector",
      //                    dimension_, "Dimension of mean vector",
      //                    &tmp);

      //   Eigen::VectorXd x_minus_mu = x - mu_;
      //   x = stan::math::mdivide_left_tri_low(L_, x_minus_mu);
      // };

    };

  } // vb

} // stan

#endif