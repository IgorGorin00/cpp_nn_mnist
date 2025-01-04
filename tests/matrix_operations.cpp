#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../include/matrix.hpp"


TEST(MatrixTest, OperatorPlus) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    Matrix result = m1 + m2;
    EXPECT_EQ(result[0][0], 6.0);
    EXPECT_EQ(result[0][1], 8.0);
    EXPECT_EQ(result[1][0], 10.0);
    EXPECT_EQ(result[1][1], 12.0);
}

TEST(MatrixTest, OperatorPlusEqual) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    m1 += m2;
    EXPECT_EQ(m1[0][0], 6.0);
    EXPECT_EQ(m1[0][1], 8.0);
    EXPECT_EQ(m1[1][0], 10.0);
    EXPECT_EQ(m1[1][1], 12.0);
}


TEST(MatrixTest, OperatorMinus) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    Matrix result = m1 - m2;
    EXPECT_EQ(result[0][0], -4.0);
    EXPECT_EQ(result[0][1], -4.0);
    EXPECT_EQ(result[1][0], -4.0);
    EXPECT_EQ(result[1][1], -4.0);
}

TEST(MatrixTest, OperatorMinusEqual) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    m1 -= m2;
    EXPECT_EQ(m1[0][0], -4.0);
    EXPECT_EQ(m1[0][1], -4.0);
    EXPECT_EQ(m1[1][0], -4.0);
    EXPECT_EQ(m1[1][1], -4.0);
}

TEST(MatrixTest, OperatorMultiply) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    Matrix result = m1 * m2;
    EXPECT_EQ(result[0][0], 5.0);
    EXPECT_EQ(result[0][1], 12.0);
    EXPECT_EQ(result[1][0], 21.0);
    EXPECT_EQ(result[1][1], 32.0);
}

TEST(MatrixTest, OperatorMultiplyEqual) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    m1 *= m2;
    EXPECT_EQ(m1[0][0], 5.0);
    EXPECT_EQ(m1[0][1], 12.0);
    EXPECT_EQ(m1[1][0], 21.0);
    EXPECT_EQ(m1[1][1], 32.0);
}


TEST(MatrixTest, OperatorDivide) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    Matrix result = m1 / m2;
    EXPECT_EQ(result[0][0], 0.2);
    EXPECT_EQ(result[0][1], 0.3333333333333333);
    EXPECT_EQ(result[1][0], 0.42857142857142855);
    EXPECT_EQ(result[1][1], 0.5);
}

TEST(MatrixTest, OperatorDivideEqual) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 6.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    m1 /= m2;
    EXPECT_EQ(m1[0][0], 0.2);
    EXPECT_EQ(m1[0][1], 0.3333333333333333);
    EXPECT_EQ(m1[1][0], 0.42857142857142855);
    EXPECT_EQ(m1[1][1], 0.5);
}


TEST(MatrixTest, CheckShapesMismatch) {
    Matrix m1(2, 2, 1.0f);
    Matrix m2(3, 3, 2.0f);
    EXPECT_THROW(m1 /= m2, std::runtime_error);
}

TEST(MatrixTest, ZeroDivision) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;

    Matrix m2(2, 2);
    m2[0][0] = 5.0; m2[0][1] = 0.0;
    m2[1][0] = 7.0; m2[1][1] = 8.0;

    EXPECT_THROW(m1 /= m2, std::runtime_error);
}


TEST(MatrixTest, OperatorPlusDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    Matrix result = m1 + x;

    EXPECT_EQ(result[0][0], 3.0);
    EXPECT_EQ(result[0][1], 4.0);
    EXPECT_EQ(result[1][0], 5.0);
    EXPECT_EQ(result[1][1], 6.0);
}

TEST(MatrixTest, OperatorPlusEqualDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    m1 += x;

    EXPECT_EQ(m1[0][0], 3.0);
    EXPECT_EQ(m1[0][1], 4.0);
    EXPECT_EQ(m1[1][0], 5.0);
    EXPECT_EQ(m1[1][1], 6.0);
}


TEST(MatrixTest, OperatorMinusDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    Matrix result = m1 - x;

    EXPECT_EQ(result[0][0], -1.0);
    EXPECT_EQ(result[0][1], 0.0);
    EXPECT_EQ(result[1][0], 1.0);
    EXPECT_EQ(result[1][1], 2.0);
}

TEST(MatrixTest, OperatorMinusEqualDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    m1 -= x;

    EXPECT_EQ(m1[0][0], -1.0);
    EXPECT_EQ(m1[0][1], 0.0);
    EXPECT_EQ(m1[1][0], 1.0);
    EXPECT_EQ(m1[1][1], 2.0);
}

TEST(MatrixTest, OperatorMultiplyDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    Matrix result = m1 * x;

    EXPECT_EQ(result[0][0], 2.0);
    EXPECT_EQ(result[0][1], 4.0);
    EXPECT_EQ(result[1][0], 6.0);
    EXPECT_EQ(result[1][1], 8.0);
}

TEST(MatrixTest, OperatorMultiplyEqualDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    m1 *= x;

    EXPECT_EQ(m1[0][0], 2.0);
    EXPECT_EQ(m1[0][1], 4.0);
    EXPECT_EQ(m1[1][0], 6.0);
    EXPECT_EQ(m1[1][1], 8.0);
}


TEST(MatrixTest, OperatorDivideDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    Matrix result = m1 / x;

    EXPECT_EQ(result[0][0], 0.5);
    EXPECT_EQ(result[0][1], 1.0);
    EXPECT_EQ(result[1][0], 1.5);
    EXPECT_EQ(result[1][1], 2.0);
}

TEST(MatrixTest, OperatorDivideEqualDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 2.0f;
    m1 /= x;

    EXPECT_EQ(m1[0][0], 0.5);
    EXPECT_EQ(m1[0][1], 1.0);
    EXPECT_EQ(m1[1][0], 1.5);
    EXPECT_EQ(m1[1][1], 2.0);
}

TEST(MatrixTest, ZeroDivisionDouble) {
    Matrix m1(2, 2);
    m1[0][0] = 1.0; m1[0][1] = 2.0;
    m1[1][0] = 3.0; m1[1][1] = 4.0;
    const double x = 0.0f;

    EXPECT_THROW(m1 /= x, std::runtime_error);
}

