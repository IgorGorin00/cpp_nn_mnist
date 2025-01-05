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

TEST(MatrixTest, RowWiseAdd) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f};

    Matrix res = m1.row_wise_add(vec);

    EXPECT_EQ(res[0][0], 11.0); EXPECT_EQ(res[0][1], 22.0); EXPECT_EQ(res[0][2], 33.0);
    EXPECT_EQ(res[1][0], 14.0); EXPECT_EQ(res[1][1], 25.0); EXPECT_EQ(res[1][2], 36.0);
    EXPECT_EQ(res[2][0], 17.0); EXPECT_EQ(res[2][1], 28.0); EXPECT_EQ(res[2][2], 39.0);
}

TEST(MatrixTest, RowWiseSubstract) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f};

    Matrix res = m1.row_wise_substract(vec);

    EXPECT_EQ(res[0][0], -9.0); EXPECT_EQ(res[0][1], -18.0); EXPECT_EQ(res[0][2], -27.0);
    EXPECT_EQ(res[1][0], -6.0); EXPECT_EQ(res[1][1], -15.0); EXPECT_EQ(res[1][2], -24.0);
    EXPECT_EQ(res[2][0], -3.0); EXPECT_EQ(res[2][1], -12.0); EXPECT_EQ(res[2][2], -21.0);
}

TEST(MatrixTest, RowWiseMultiply) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f};

    Matrix res = m1.row_wise_multiply(vec);

    EXPECT_EQ(res[0][0], 10.0); EXPECT_EQ(res[0][1], 40.0); EXPECT_EQ(res[0][2], 90.0);
    EXPECT_EQ(res[1][0], 40.0); EXPECT_EQ(res[1][1], 100.0); EXPECT_EQ(res[1][2], 180.0);
    EXPECT_EQ(res[2][0], 70.0); EXPECT_EQ(res[2][1], 160.0); EXPECT_EQ(res[2][2], 270.0);
}

TEST(MatrixTest, RowWiseDivide) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f};

    Matrix res = m1.row_wise_divide(vec);

    EXPECT_EQ(res[0][0], 0.1); EXPECT_EQ(res[0][1], 0.1); EXPECT_EQ(res[0][2], 0.1);
    EXPECT_EQ(res[1][0], 0.4); EXPECT_EQ(res[1][1], 0.25); EXPECT_EQ(res[1][2], 0.2);
    EXPECT_EQ(res[2][0], 0.7); EXPECT_EQ(res[2][1], 0.4); EXPECT_EQ(res[2][2], 0.3);
}

TEST(MatrixTest, RowWiseAddShapeFail) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f, 40.0f};

    EXPECT_THROW(m1.row_wise_add(vec), std::runtime_error);
}

TEST(MatrixTest, RowWiseDivideZeroFail) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 0.0f, 30.0f};

    EXPECT_THROW(m1.row_wise_divide(vec), std::runtime_error);
}

TEST(MatrixTest, ColWiseAdd) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 20.0f, 30.0f};

    Matrix res = m1.col_wise_add(vec);

    EXPECT_EQ(res[0][0], 11.0); EXPECT_EQ(res[0][1], 12.0); EXPECT_EQ(res[0][2], 13.0);
    EXPECT_EQ(res[1][0], 24.0); EXPECT_EQ(res[1][1], 25.0); EXPECT_EQ(res[1][2], 26.0);
    EXPECT_EQ(res[2][0], 37.0); EXPECT_EQ(res[2][1], 38.0); EXPECT_EQ(res[2][2], 39.0);
}

TEST(MatrixTest, ColWiseSubstract) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{
            10.0f, 
            20.0f, 
            30.0f};

    Matrix res = m1.col_wise_substract(vec);

    EXPECT_EQ(res[0][0], -9.0); EXPECT_EQ(res[0][1], -8.0); EXPECT_EQ(res[0][2], -7.0);
    EXPECT_EQ(res[1][0], -16.0); EXPECT_EQ(res[1][1], -15.0); EXPECT_EQ(res[1][2], -14.0);
    EXPECT_EQ(res[2][0], -23.0); EXPECT_EQ(res[2][1], -22.0); EXPECT_EQ(res[2][2], -21.0);
}

TEST(MatrixTest, ColWiseMultiply) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{
            10.0f, 
            20.0f, 
            30.0f};

    Matrix res = m1.col_wise_multiply(vec);

    EXPECT_EQ(res[0][0], 10.0); EXPECT_EQ(res[0][1], 20.0); EXPECT_EQ(res[0][2], 30.0);
    EXPECT_EQ(res[1][0], 80.0); EXPECT_EQ(res[1][1], 100.0); EXPECT_EQ(res[1][2], 120.0);
    EXPECT_EQ(res[2][0], 210.0); EXPECT_EQ(res[2][1], 240.0); EXPECT_EQ(res[2][2], 270.0);
}

TEST(MatrixTest, ColWiseDivide) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{
            10.0f, 
            20.0f, 
            30.0f};

    Matrix res = m1.col_wise_divide(vec);

    EXPECT_EQ(res[0][0], 0.1); EXPECT_EQ(res[0][1], 0.2); EXPECT_EQ(res[0][2], 0.3);
    EXPECT_EQ(res[1][0], 0.2); EXPECT_EQ(res[1][1], 0.25); EXPECT_EQ(res[1][2], 0.3);
    EXPECT_EQ(res[2][0], 0.23333333333333334); EXPECT_EQ(res[2][1], 0.26666666666666666); EXPECT_EQ(res[2][2], 0.3);
}

TEST(MatrixTest, ColWiseDivideZeroFail) {
    Matrix m1(3, 3);
    m1[0][0] = 1.0; m1[0][1] = 2.0; m1[0][2] = 3.0;
    m1[1][0] = 4.0; m1[1][1] = 5.0; m1[1][2] = 6.0;
    m1[2][0] = 7.0; m1[2][1] = 8.0; m1[2][2] = 9.0;

    std::vector<double> vec{10.0f, 0.0f, 30.0f};

    EXPECT_THROW(m1.col_wise_divide(vec), std::runtime_error);
}

TEST(MatrixTest, Dot) {
    Matrix m1(2, 3);
    m1[0][0] = 2.0; m1[0][1] = 3.0; m1[0][2] = 4.0;
    m1[1][0] = 1.0; m1[1][1] = 0.0; m1[1][2] = 0.0;

    Matrix m2(3, 2);
    m2[0][0] = 0.0; m2[0][1] = 1000.0;
    m2[1][0] = 1.0; m2[1][1] = 100.0;
    m2[2][0] = 0.0; m2[2][1] = 10.0;

    Matrix result = m1.dot(m2);
    EXPECT_EQ(result[0][0], 3.0); EXPECT_EQ(result[0][1], 2340.0);
    EXPECT_EQ(result[1][0], 0.0); EXPECT_EQ(result[1][1], 1000.0);
}

TEST(MatrixTest, DotLarge) {
    Matrix m1(30, 10);
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 10; j++) {
            m1[i][j] = i + j;
        }
    }

    Matrix m2(10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            m2[i][j] = i * j;
        }
    }

    Matrix result = m1.dot(m2);
    for (int i = 0; i < 30; i++) {
        for (int j = 0; j < 10; j++) {
            double expected = 0;
            for (int k = 0; k < 10; k++) {
                expected += (i + k) * (k * j);
            }
            EXPECT_NEAR(result[i][j], expected, 1e-6);
        }
    }
}

TEST(MatrixTest, DotShapeMismatch) {
    Matrix m1(5, 3);
    Matrix m2(5, 2);
    EXPECT_THROW(m1.dot(m2), std::runtime_error);
}
