#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>

using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::Array;

TEST(ProbBernoulliLogit, log_matches_lpmf) {
  Matrix<int, Dynamic, 1> n(3, 1);
  n << 0, 1, 0;
  Matrix<double, Dynamic, 1> theta(3, 1);
  theta << 1.2, 2, 0.9;

  EXPECT_FLOAT_EQ((stan::math::bernoulli_logit_lpmf(n, theta)),
                  (stan::math::bernoulli_logit_log(n, theta)));
  EXPECT_FLOAT_EQ((stan::math::bernoulli_logit_lpmf<true>(n, theta)),
                  (stan::math::bernoulli_logit_log<true>(n, theta)));
  EXPECT_FLOAT_EQ((stan::math::bernoulli_logit_lpmf<false>(n, theta)),
                  (stan::math::bernoulli_logit_log<false>(n, theta)));
  EXPECT_FLOAT_EQ(
      (stan::math::bernoulli_logit_lpmf<true, Matrix<int, Dynamic, 1>>(n,
                                                                       theta)),
      (stan::math::bernoulli_logit_log<true, Matrix<int, Dynamic, 1>>(n,
                                                                      theta)));
  EXPECT_FLOAT_EQ(
      (stan::math::bernoulli_logit_lpmf<false, Matrix<int, Dynamic, 1>>(n,
                                                                        theta)),
      (stan::math::bernoulli_logit_log<false, Matrix<int, Dynamic, 1>>(n,
                                                                       theta)));
  EXPECT_FLOAT_EQ(
      (stan::math::bernoulli_logit_lpmf<Matrix<int, Dynamic, 1>>(n, theta)),
      (stan::math::bernoulli_logit_log<Matrix<int, Dynamic, 1>>(n, theta)));
}


TEST(ProbBernoulliLogit, test_different_types) {
  std::vector<double> vec_test(3);
  vec_test[0] = 0; vec_test[1] = 1; vec_test[2] = 2;
  Matrix<int, Dynamic, 1> n(3, 1);
  n << 0, 1, 0;
  EXPECT_NO_THROW(stan::math::bernoulli_logit_lpmf(n, vec_test));

  double scalar_test = 1.0;
  int n1 = 1.0;
  EXPECT_NO_THROW(stan::math::bernoulli_logit_lpmf(n1, scalar_test));

  Matrix<double, 1, -1> row_vec(3);
  row_vec << 1, 2, 3;
  EXPECT_NO_THROW(stan::math::bernoulli_logit_lpmf(n, row_vec));
}
