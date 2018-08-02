#include <gtest/gtest.h>
#include <stan/math/rev/mat.hpp>
#include <limits>
#include <string>
#include <vector>

template <typename T_x, typename T_sigma>
std::string pull_msg(std::vector<T_x> x, T_sigma sigma) {
  std::string message;
  try {
    stan::math::gp_dot_prod_cov(x, sigma);
  } catch (std::domain_error &e) {
    message = e.what();
  } catch (...) {
    message = "Threw the wrong exection";
  }
  return message;
}

template <typename T_x1, typename T_x2, typename T_sigma>
std::string pull_msg(std::vector<T_x1> x1, std::vector<T_x2> x2,
                     T_sigma sigma) {
  std::string message;
  try {
    stan::math::gp_dot_prod_cov(x1, x2, sigma);
  } catch (std::domain_error &e) {
    message = e.what();
  } catch (...) {
    message = "Threw the wrong exection";
  }
  return message;
}

TEST(RevMath, gp_dot_prod_cov_vvv) {
  std::vector<stan::math::var> x(3);
  stan::math::var sigma = 1.5;
  x[0] = -2;
  x[1] = -1;
  x[2] = -0.5;

  Eigen::Matrix<stan::math::var, -1, -1> cov;
  EXPECT_NO_THROW(cov = stan::math::gp_dot_prod_cov(x, sigma));

  std::vector<double> grad;
  std::vector<stan::math::var> params;
  params.push_back(sigma);
  params.push_back(x[0]);
  params.push_back(x[1]);
  params.push_back(x[2]);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // cov(i, j).grad(params, grad);
      // EXPECT_FLOAT_EQ(stan::math::square(sigma.val()) +
      //                 x[i].val() * x[j].val(),
      //                 stan::math::value_of(cov(i, j)))
      //   << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(2 * sigma.val(), grad[0])
      //   << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(1.0, grad[1])
      //   << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(1.0, grad[2])
      //   << "index: (" << i << ", " << j << ")";
      // EXPECT_FLOAT_EQ(1.0, grad[3])
      //   << "index: (" << i << ", " << j << ")";
    }
  }
  stan::math::recover_memory();
}
