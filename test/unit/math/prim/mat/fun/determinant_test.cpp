#include <stan/math/prim/mat.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, dimensionValidation) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::determinant;
  Matrix<double, Dynamic, Dynamic> x(3, 3);
  x << 1, 2, 3, 1, 4, 9, 1, 8, 27;

  ASSERT_FALSE(boost::math::isnan(determinant(x)));

  Matrix<double, Dynamic, Dynamic> xx(3, 2);
  xx << 1, 2, 3, 1, 4, 9;
  EXPECT_THROW(stan::math::determinant(xx), std::invalid_argument);
}
