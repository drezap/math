#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/util.hpp>
#include <stan/math/rev/mat/fun/gp_exponential_cov.hpp>
//#include <stan/math/prim/mat/fun/gp_exponential_cov.hpp>
#include <stan/math/prim/mat/fun/divide_columns.hpp>
#include <limits>
#include <string>
#include <vector>

template <typename T_x1, typename T_x2, typename T_sigma, typename T_l>
std::string pull_msg(std::vector<T_x1> x1, std::vector<T_x2> x2, T_sigma sigma,
                     T_l l) {
  std::string message;
  try {
    stan::math::gp_exp_quad_cov(x1, x2, sigma, l);
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    message = "Threw the wrong exception";
  }
  return message;
}

template <typename T_x1, typename T_sigma, typename T_l>
std::string pull_msg(std::vector<T_x1> x1, T_sigma sigma, T_l l) {
  std::string message;
  try {
    stan::math::gp_exp_quad_cov(x1, sigma, l);
  } catch (std::domain_error& e) {
    message = e.what();
  } catch (...) {
    message = "Threw the wrong exception";
  }
  return message;
}

TEST(RevMath, gp_exponential_cov_vvv) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<stan::math::var> x(3);
      stan::math::var sigma = 0.2;
      stan::math::var l = 5;
      x[0] = -2;
      x[1] = -1;
      x[2] = -0.5;
      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);
      params.push_back(l);
      params.push_back(x[i]);
      params.push_back(x[j]);

      cov(i, j).grad(params, grad);
      double dist = std::abs(x[i].val() - x[j].val());
      double exp_val = exp(-dist / l.val());
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(
          sigma.val() * sigma.val() * exp_val * dist / (l.val() * l.val()),
          grad[1])
          << "index: (" << i << ", " << j << ")";
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vvd) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  double l = 5;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<stan::math::var> x(3);
      stan::math::var sigma = 0.2;
      x[0] = -2;
      x[1] = -1;
      x[2] = -0.5;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);
      params.push_back(x[i]);
      params.push_back(x[j]);

      cov(i, j).grad(params, grad);
      double dist = std::abs(x[i].val() - x[j].val());
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";
      if (x[i] < x[j]) {
        EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * exp_val / l, grad[1])
            << "index: (" << i << ", " << j << ")";
      } else if (x[i] > x[j]) {
        EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * -exp_val / l, grad[1])
            << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
      }

      if (x[i] > x[j]) {
        EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * exp_val / l, grad[2])
            << "index: (" << i << ", " << j << ")";
      } else if (x[i] < x[j]) {
        EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * -exp_val / l, grad[2])
            << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
      }

      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vdv) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  double sigma = 0.2;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<stan::math::var> x(3);
      stan::math::var l = 5;
      x[0] = -2;
      x[1] = -1;
      x[2] = -0.5;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(l);
      params.push_back(x[i]);
      params.push_back(x[j]);

      cov(i, j).grad(params, grad);
      double dist = std::abs(x[i].val() - x[j].val());
      double exp_val = exp(-dist / l.val());
      EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(sigma * sigma * exp_val * dist / (l.val() * l.val()),
                      grad[0])
          << "index: (" << i << ", " << j << ")";
      if (x[i] < x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * exp_val / l.val(), grad[1])
            << "index: (" << i << ", " << j << ")";
      } else if (x[i] > x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * -exp_val / l.val(), grad[1])
            << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
      }

      if (x[i] > x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * exp_val / l.val(), grad[2])
            << "index: (" << i << ", " << j << ")";
      } else if (x[i] < x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * -exp_val / l.val(), grad[2])
            << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
      }

      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vdd) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  double sigma = 0.2;
  double l = 5;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<stan::math::var> x(3);
      x[0] = -2;
      x[1] = -1;
      x[2] = -0.5;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(x[i]);
      params.push_back(x[j]);

      cov(i, j).grad(params, grad);
      double dist = std::abs(x[i].val() - x[j].val());
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      if (x[i] < x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * -exp_val / l, grad[1])
            << "index: (" << i << ", " << j << ")";
      } else if (x[i] > x[j]) {
        EXPECT_FLOAT_EQ(sigma * sigma * exp_val / l, grad[1])
            << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
      }
      stan::math::recover_memory();
    }
  }
}

// TEST(RevMath, gp_exponential_cov_dvv) {
//   Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
//   std::vector<double> x(3);
//   x[0] = -2;
//   x[1] = -1;
//   x[2] = -0.5;

//   for (std::size_t i = 0; i < 3; ++i) {
//     for (std::size_t j = 0; j < 3; ++j) {
//       stan::math::var sigma = 0.2;
//       stan::math::var l = 5;

//       EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

//       std::vector<double> grad;
//       std::vector<stan::math::var> params;
//       params.push_back(sigma);
//       params.push_back(l);

//       cov(i, j).grad(params, grad);

//       double dist = std::abs(x[i] - x[j]);
//       double exp_val = exp(-dist / l.val());
//       EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
//                       cov(i, j).val())
//           << "index: (" << i << ", " << j << ")";
//       EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
//           << "index: (" << i << ", " << j << ")";
//       EXPECT_FLOAT_EQ(
//           sigma.val() * sigma.val() * exp_val * dist / (l.val() *
// l.val()),
//           grad[1])
//           << "index: (" << i << ", " << j << ")";
//       stan::math::recover_memory();
//     }
//   }
// }

TEST(RevMath, gp_exponential_cov_dvd) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  std::vector<double> x(3);
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;
  double l = 5.0;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      stan::math::var sigma = 0.2;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);

      cov(i, j).grad(params, grad);

      double dist = std::abs(x[i] - x[j]);
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";

      stan::math::recover_memory();
    }
  }
}

// TEST(RevMath, gp_exponential_cov_ddv) {
//   Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
//   std::vector<double> x(3);
//   x[0] = -2;
//   x[1] = -1;
//   x[2] = -0.5;
//   double sigma = 0.2;

//   for (std::size_t i = 0; i < 3; ++i) {
//     for (std::size_t j = 0; j < 3; ++j) {
//       stan::math::var l = 5;

//       EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

//       std::vector<double> grad;
//       std::vector<stan::math::var> params;
//       params.push_back(l);

//       cov(i, j).grad(params, grad);

//       double dist = std::abs(x[i] - x[j]);
//       double exp_val = exp(-dist / l.val());
//       EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
//           << "index: (" << i << ", " << j << ")";
//       EXPECT_FLOAT_EQ(sigma * sigma * exp_val * dist / (l.val() * l.val()),
//                       grad[0]);
//       stan::math::recover_memory();
//     }
//   }
// }

TEST(RevMath, gp_exponential_cov_vector_vvv) {
  typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vector_v;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<vector_v> x(3);
      vector_v x0(2), x1(2), x2(2);
      x0(0) = -2;
      x0(1) = -2;
      x1(0) = 1;
      x1(1) = 2;
      x2(0) = -0.5;
      x2(1) = 0.0;

      x[0] = x0;
      x[1] = x1;
      x[2] = x2;

      stan::math::var sigma = 0.2;
      stan::math::var l = 5.0;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);
      params.push_back(l);
      params.push_back(x[i](0));
      params.push_back(x[i](1));
      params.push_back(x[j](0));
      params.push_back(x[j](1));

      cov(i, j).grad(params, grad);

      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                         stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l.val());
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(
          sigma.val() * sigma.val() * exp_val * dist / (l.val() * l.val()),
          grad[1])
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_vvd) {
  typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vector_v;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  double l = 5.0;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<vector_v> x(3);
      vector_v x0(2), x1(2), x2(2);
      x0(0) = -2;
      x0(1) = -2;
      x1(0) = 1;
      x1(1) = 2;
      x2(0) = -0.5;
      x2(1) = 0.0;

      x[0] = x0;
      x[1] = x1;
      x[2] = x2;

      stan::math::var sigma = 0.2;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);
      params.push_back(x[i](0));
      params.push_back(x[i](1));
      params.push_back(x[j](0));
      params.push_back(x[j](1));

      cov(i, j).grad(params, grad);
      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                         stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";
      if (i == j) {
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[3]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[4]) << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val * (1 / l) * -1
                            / dist * (x[i](0).val() - x[j](0).val()),
                        grad[1])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val * (1 / l) * -1
                            / dist * (x[i](1).val() - x[j](1).val()),
                        grad[2])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val * (1 / l) * -1
                            / dist * (x[j](0).val() - x[i](0).val()),
                        grad[3])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val * (1 / l) * -1
                            / dist * (x[j](1).val() - x[i](1).val()),
                        grad[4])
            << "index: (" << i << ", " << j << ")";
      }
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_vdv) {
  typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vector_v;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  double sigma = 0.2;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<vector_v> x(3);
      vector_v x0(2), x1(2), x2(2);
      x0(0) = -2;
      x0(1) = -2;
      x1(0) = 1;
      x1(1) = 2;
      x2(0) = -0.5;
      x2(1) = 0.0;

      x[0] = x0;
      x[1] = x1;
      x[2] = x2;

      stan::math::var l = 5;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(l);
      params.push_back(x[i](0));
      params.push_back(x[i](1));
      params.push_back(x[j](0));
      params.push_back(x[j](1));

      cov(i, j).grad(params, grad);

      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                         stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l.val());
      EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      if (i == j) {
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[3]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[4]) << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l.val()) * -1
                            / dist * (x[i](0).val() - x[j](0).val()),
                        grad[1])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l.val()) * -1
                            / dist * (x[i](1).val() - x[j](1).val()),
                        grad[2])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l.val()) * -1
                            / dist * (x[j](0).val() - x[i](0).val()),
                        grad[3])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l.val()) * -1
                            / dist * (x[j](1).val() - x[i](1).val()),
                        grad[4])
            << "index: (" << i << ", " << j << ")";
      }
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_vdd) {
  typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vector_v;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  double sigma = 0.2;
  double l = 5;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::vector<vector_v> x(3);
      vector_v x0(2), x1(2), x2(2);
      x0(0) = -2;
      x0(1) = -2;
      x1(0) = 1;
      x1(1) = 2;
      x2(0) = -0.5;
      x2(1) = 0.0;

      x[0] = x0;
      x[1] = x1;
      x[2] = x2;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(x[i](0));
      params.push_back(x[i](1));
      params.push_back(x[j](0));
      params.push_back(x[j](1));

      cov(i, j).grad(params, grad);

      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                         stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      if (i == j) {
        EXPECT_FLOAT_EQ(0, grad[0]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[1]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[2]) << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(0, grad[3]) << "index: (" << i << ", " << j << ")";
      } else {
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l) * -1
                            / dist * (x[i](0).val() - x[j](0).val()),
                        grad[0])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l) * -1
                            / dist * (x[i](1).val() - x[j](1).val()),
                        grad[1])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l) * -1
                            / dist * (x[j](0).val() - x[i](0).val()),
                        grad[2])
            << "index: (" << i << ", " << j << ")";
        EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val * (1 / l) * -1
                            / dist * (x[j](1).val() - x[i](1).val()),
                        grad[3])
            << "index: (" << i << ", " << j << ")";
      }
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_dvv) {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_d;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  std::vector<vector_d> x(3);
  vector_d x0(2), x1(2), x2(2);
  x0(0) = -2;
  x0(1) = -2;
  x1(0) = 1;
  x1(1) = 2;
  x2(0) = -0.5;
  x2(1) = 0.0;

  x[0] = x0;
  x[1] = x1;
  x[2] = x2;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      stan::math::var sigma = 0.2;
      stan::math::var l = 5;

      // won't work until we write a specialization
      //      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma,
      //      l));

      // std::vector<double> grad;
      // std::vector<stan::math::var> params;
      // params.push_back(sigma);
      // params.push_back(l);

      // cov(i, j).grad(params, grad);
      // double dist = stan::math::distance(
      //     stan::math::value_of(x[i]), stan::math::value_of(x[j]));
      // double exp_val = exp(-dist / l.val());
      // EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
      //                 cov(i, j).val())
      //     << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
      //     << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * exp_val / l.val(), grad[1])
      //     << "index: (" << i << ", " << j << ")";
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_dvd) {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_d;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  std::vector<vector_d> x(3);
  vector_d x0(2), x1(2), x2(2);
  x0(0) = -2;
  x0(1) = -2;
  x1(0) = 1;
  x1(1) = 2;
  x2(0) = -0.5;
  x2(1) = 0.0;

  x[0] = x0;
  x[1] = x1;
  x[2] = x2;
  double l = 5.0;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      stan::math::var sigma = 0.2;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma,
                                                           l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(sigma);

      cov(i, j).grad(params, grad);

      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                             stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l);
      EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) * exp_val,
                      cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(2 * sigma.val() * exp_val, grad[0])
          << "index: (" << i << ", " << j << ")";
      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_vector_ddv) {
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_d;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;
  std::vector<vector_d> x(3);
  vector_d x0(2), x1(2), x2(2);
  x0(0) = -2;
  x0(1) = -2;
  x1(0) = 1;
  x1(1) = 2;
  x2(0) = -0.5;
  x2(1) = 0.0;

  x[0] = x0;
  x[1] = x1;
  x[2] = x2;
  double sigma = 0.2;

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      stan::math::var l = 5.0;

      EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

      std::vector<double> grad;
      std::vector<stan::math::var> params;
      params.push_back(l);

      cov(i, j).grad(params, grad);

      double dist = stan::math::distance(stan::math::value_of(x[i]),
                                             stan::math::value_of(x[j]));
      double exp_val = exp(-dist / l.val());
      EXPECT_FLOAT_EQ(stan::math::square(sigma) * exp_val, cov(i, j).val())
          << "index: (" << i << ", " << j << ")";
      EXPECT_FLOAT_EQ(sigma * sigma * exp_val * dist /(l.val() *
      l.val()),
                      grad[0])
          << "index: (" << i << ", " << j << ")";

      stan::math::recover_memory();
    }
  }
}

TEST(RevMath, gp_exponential_cov_domain_error_training) {
  using stan::math::var;
  var sigma = 0.2;
  var l = 5;

  std::vector<var> x(3);
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;

  std::vector<Eigen::Matrix<var, -1, 1> > x_2(3);
  for (size_t i = 0; i < x_2.size(); ++i) {
    x_2[i].resize(3, 1);
    x_2[i] << 1, 2, 3;
  }

  var sigma_bad = -1;
  var l_bad = -1;

  std::string msg1, msg2, msg3;
  msg1 = pull_msg(x, sigma, l_bad);
  msg2 = pull_msg(x, sigma_bad, l);
  msg3 = pull_msg(x, sigma_bad, l_bad);
  EXPECT_TRUE(std::string::npos != msg1.find(" length scale")) << msg1;
  EXPECT_TRUE(std::string::npos != msg2.find(" magnitude")) << msg2;
  EXPECT_TRUE(std::string::npos != msg3.find(" magnitude")) << msg3;

  EXPECT_THROW(stan::math::gp_exponential_cov(x, sigma, l_bad),
  std::domain_error); EXPECT_THROW(stan::math::gp_exponential_cov(x, sigma_bad,
  l), std::domain_error); EXPECT_THROW(stan::math::gp_exponential_cov(x,
  sigma_bad, l_bad),
               std::domain_error);

  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma_bad, l_bad),
               std::domain_error);
}

TEST(RevMath, gp_exponential_cov_nan_error_training) {
  using stan::math::var;
  var sigma = 0.2;
  var l = 5;

  std::vector<var> x(3);
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;

  std::vector<Eigen::Matrix<var, -1, 1> > x_2(3);
  for (size_t i = 0; i < x_2.size(); ++i) {
    x_2[i].resize(3, 1);
    x_2[i] << 1, 2, 3;
  }

  std::vector<var> x_bad(x);
  x_bad[1] = std::numeric_limits<var>::quiet_NaN();

  std::vector<Eigen::Matrix<var, -1, 1> > x_bad_2(x_2);
  x_bad_2[1](1) = std::numeric_limits<var>::quiet_NaN();

  var sigma_bad = std::numeric_limits<var>::quiet_NaN();
  var l_bad = std::numeric_limits<var>::quiet_NaN();

  std::string msg1, msg2, msg3;
  msg1 = pull_msg(x, sigma, l_bad);
  msg2 = pull_msg(x, sigma_bad, l);
  msg3 = pull_msg(x, sigma_bad, l_bad);
  EXPECT_TRUE(std::string::npos != msg1.find(" length scale")) << msg1;
  EXPECT_TRUE(std::string::npos != msg2.find(" magnitude")) << msg2;
  EXPECT_TRUE(std::string::npos != msg3.find(" magnitude")) << msg3;

  EXPECT_THROW(stan::math::gp_exponential_cov(x, sigma, l_bad),
  std::domain_error); EXPECT_THROW(stan::math::gp_exponential_cov(x, sigma_bad,
  l), std::domain_error); EXPECT_THROW(stan::math::gp_exponential_cov(x,
  sigma_bad, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad, sigma, l),
  std::domain_error); EXPECT_THROW(stan::math::gp_exponential_cov(x_bad,
  sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad, sigma_bad, l_bad),
               std::domain_error);

  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_2, sigma_bad, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad_2, sigma, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad_2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad_2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_bad_2, sigma_bad, l_bad),
               std::domain_error);
}

TEST(RevMath, gp_exponential_cov_domain_error) {
  using stan::math::var;
  var sigma = 0.2;
  var l = 5;

  std::vector<var> x1(3);
  x1[0] = -2;
  x1[1] = -1;
  x1[2] = -0.5;

  std::vector<var> x2(4);
  x2[0] = -2;
  x2[1] = -1;
  x2[2] = -0.5;
  x2[3] = -5;

  var sigma_bad = -1;
  var l_bad = -1;

  std::string msg1, msg2, msg3;
  msg1 = pull_msg(x1, x2, sigma, l_bad);
  msg2 = pull_msg(x1, x2, sigma_bad, l);
  msg3 = pull_msg(x1, x2, sigma_bad, l_bad);
  EXPECT_TRUE(std::string::npos != msg1.find(" length scale")) << msg1;
  EXPECT_TRUE(std::string::npos != msg2.find(" magnitude")) << msg2;
  EXPECT_TRUE(std::string::npos != msg3.find(" magnitude")) << msg3;

  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma_bad, l_bad),
               std::domain_error);

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_1(3);
  for (size_t i = 0; i < x_vec_1.size(); ++i) {
    x_vec_1[i].resize(3, 1);
    x_vec_1[i] << 1, 2, 3;
  }

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_2(4);
  for (size_t i = 0; i < x_vec_2.size(); ++i) {
    x_vec_2[i].resize(3, 1);
    x_vec_2[i] << 4, 1, 3;
  }

  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma_bad,
  l_bad),
               std::domain_error);
}

TEST(RevMath, gp_exponential_cov2_nan_domain_error) {
  using stan::math::var;
  var sigma = 0.2;
  var l = 5;

  std::vector<var> x1(3);
  x1[0] = -2;
  x1[1] = -1;
  x1[2] = -0.5;

  std::vector<var> x2(4);
  x2[0] = -2;
  x2[1] = -1;
  x2[2] = -0.5;
  x2[3] = -5;

  var sigma_bad = std::numeric_limits<var>::quiet_NaN();
  var l_bad = std::numeric_limits<var>::quiet_NaN();

  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2, sigma_bad, l_bad),
               std::domain_error);

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_1(3);
  for (size_t i = 0; i < x_vec_1.size(); ++i) {
    x_vec_1[i].resize(3, 1);
    x_vec_1[i] << 1, 2, 3;
  }

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_2(4);
  for (size_t i = 0; i < x_vec_2.size(); ++i) {
    x_vec_2[i].resize(3, 1);
    x_vec_2[i] << 4, 1, 3;
  }

  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma_bad, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma, l_bad),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma_bad,
  l_bad),
               std::domain_error);

  std::vector<var> x1_bad(x1);
  x1_bad[1] = std::numeric_limits<var>::quiet_NaN();
  std::vector<var> x2_bad(x2);
  x2_bad[1] = std::numeric_limits<var>::quiet_NaN();

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_1_bad(x_vec_1);
  x_vec_1_bad[1](1) = std::numeric_limits<var>::quiet_NaN();
  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_2_bad(x_vec_2);
  x_vec_2_bad[1](1) = std::numeric_limits<var>::quiet_NaN();

  EXPECT_THROW(stan::math::gp_exponential_cov(x1_bad, x2, sigma, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1, x2_bad, sigma, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x1_bad, x2_bad, sigma, l),
               std::domain_error);

  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1_bad, x_vec_2, sigma, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2_bad, sigma, l),
               std::domain_error);
  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1_bad, x_vec_2_bad, sigma,
  l),
               std::domain_error);
}

TEST(RevMath, gp_exponential_cov2_dim_mismatch_vec_eigen_vec) {
  using stan::math::var;
  var sigma = 0.2;
  var l = 5;

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_1(3);
  for (size_t i = 0; i < x_vec_1.size(); ++i) {
    x_vec_1[i].resize(3, 1);
    x_vec_1[i] << 1, 2, 3;
  }

  std::vector<Eigen::Matrix<var, -1, 1> > x_vec_2(4);
  for (size_t i = 0; i < x_vec_2.size(); ++i) {
    x_vec_2[i].resize(4, 1);
    x_vec_2[i] << 4, 1, 3, 1;
  }

  EXPECT_THROW(stan::math::gp_exponential_cov(x_vec_1, x_vec_2, sigma, l),
               std::invalid_argument);
}

TEST(AgradRevMatrix, check_varis_on_stack) {
  using stan::math::to_var;
  std::vector<double> x(3);
  double sigma = 0.2;
  double l = 5;
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;

  test::check_varis_on_stack(
      stan::math::gp_exp_quad_cov(to_var(x), to_var(sigma), to_var(l)));
  test::check_varis_on_stack(
      stan::math::gp_exp_quad_cov(to_var(x), to_var(sigma), l));
  test::check_varis_on_stack(
      stan::math::gp_exp_quad_cov(to_var(x), sigma, to_var(l)));
  test::check_varis_on_stack(stan::math::gp_exp_quad_cov(to_var(x), sigma,
  l)); test::check_varis_on_stack(
      stan::math::gp_exp_quad_cov(x, to_var(sigma), to_var(l)));
  test::check_varis_on_stack(stan::math::gp_exp_quad_cov(x, to_var(sigma),
  l)); test::check_varis_on_stack(stan::math::gp_exp_quad_cov(x, sigma,
  to_var(l)));
}

TEST(RevMath, gp_exponential_cov_ard) {
  //  typedef Eigen::Matrix<stan::math::var, Eigen::Dynamic, 1> vector_v;
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> cov;

  stan::math::var sigma = 1.2;
  std::vector<stan::math::var> l(3);
  l[0] = 1.0; l[1] = 2.0; l[2] = 3.0;
  
  std::vector<Eigen::Matrix<double, -1, 1>> x(3);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i].resize(3, 1);
    x[i] << 1 * i, 2 * i, 3 * i;
  }
  EXPECT_NO_THROW(cov = stan::math::gp_exponential_cov(x, sigma, l));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {

      std::vector<Eigen::Matrix<stan::math::var, -1, 1>> x_new =
        stan::math::divide_columns(x, l);
      double dist = stan::math::distance(x_new[i], x_new[j]).val();
      double exp_val = exp(-dist);
      EXPECT_FLOAT_EQ(sigma.val() * sigma.val() * exp_val,
                      cov(i, j).val())
        << "index: (" << i << ", " << j << ")";
    }
  }
  
  EXPECT_NO_THROW(stan::math::gp_exponential_cov(x, sigma, l));
}
