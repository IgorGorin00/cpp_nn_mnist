#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../include/matrix.hpp"

TEST(MatrixTest, ConstructorWithMeanAndScale) {
    Matrix matrix(3, 3, 1.0, 2.0);
    EXPECT_EQ(matrix.nRows, 3);
    EXPECT_EQ(matrix.nCols, 3);
}

TEST(MatrixTest, ConstructorWithMean) {
    Matrix matrix(5, 5, 1.0);
    EXPECT_EQ(matrix.nRows, 5);
    EXPECT_EQ(matrix.nCols, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_EQ(matrix.vals[i][j], 1.0f);
        }
    }
}

TEST(MatrixTest, ConstructorWithNoMean) {
    Matrix matrix(5, 5);
    EXPECT_EQ(matrix.nRows, 5);
    EXPECT_EQ(matrix.nCols, 5);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            EXPECT_EQ(matrix.vals[i][j], 0.0f);
        }
    }
}

TEST(MatrixTest, ConstructorWithPreVals) {
    std::vector<std::vector<double>> pre_vals = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Matrix matrix(pre_vals);
    EXPECT_EQ(matrix.nRows, 3);
    EXPECT_EQ(matrix.nCols, 3);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(matrix.vals[i][j], pre_vals[i][j]);
        }
    }
}

TEST(MatrixTest, Print) {
    Matrix matrix(3, 3, 1.0, 2.0);
    // You can't directly test the output of print(), 
    // but you can test that it doesn't crash.
    matrix.print();
}

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
// int main(int argc, char* argv[]) {
//     Matrix mat(4, 4);
//     mat.print();
// }


