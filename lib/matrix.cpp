#include "../include/matrix.hpp"
#include <cstdio>
#include <random>
#include <stdexcept>

Matrix::Matrix(
        const int n_rows_, const int n_cols_, 
        const double mean, const double scale
        ) :
    nRows(n_rows_),
    nCols(n_cols_),
    vals(n_rows_, std::vector<double>(n_cols_, mean)) {
        if (scale != 0.0f) {
            std::random_device rd{};
            std::mt19937 gen(rd());
            std::normal_distribution<double> d{mean, scale};
            for (int i = 0; i < n_rows_; i++) {
                for (int j = 0; j < n_cols_; j++) {
                    this->vals[i][j] = d(gen);

                }
            }
        }

    };

Matrix::Matrix(std::vector<std::vector<double>>& pre_vals) 
    : nRows(pre_vals.size()), nCols(pre_vals[0].size()), vals(pre_vals) {};

void Matrix::print() const {
    for (const std::vector<double>& row : this->vals) {
        printf("{ ");
        for (const double elem : row) {
            printf("%f ", elem);
        }
        printf(" }\n");
    }
}

std::vector<double>& Matrix::operator[](const int idx) {
    return this->vals[idx];
}; 

const std::vector<double>& Matrix::operator[](const int idx) const {
    return this-> vals[idx];
}; 

void Matrix::check_shapes_mismatch(const Matrix& other) const {
    if ((this->nRows != other.nRows) || (this->nCols != other.nCols)) {
        printf("this->shape = (%d, %d) other.shape = (%d, %d)\n", 
                this->nRows, this->nCols, other.nRows, other.nCols);
        throw std::runtime_error("Matrix::operator+-*= shape mismatch!\n"
                                 "Shapes should be the same!\n");
    } 
};

Matrix Matrix::operator+(const Matrix& other) const {
    this->check_shapes_mismatch(other);
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] += other[i][j];
        }
    }
    return res;
}; 

Matrix Matrix::operator+=(const Matrix& other) {
    this->check_shapes_mismatch(other);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] += other[i][j];
        }
    }
    return *this;
}; 

Matrix Matrix::operator-(const Matrix& other) const {
    this->check_shapes_mismatch(other);
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] -= other[i][j];
        }
    }
    return res;
}; 

Matrix Matrix::operator-=(const Matrix& other) {
    this->check_shapes_mismatch(other);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] -= other[i][j];
        }
    }
    return *this;
}; 

Matrix Matrix::operator*(const Matrix& other) const {
    this->check_shapes_mismatch(other);
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] *= other[i][j];
        }
    }
    return res;
}; 

Matrix Matrix::operator*=(const Matrix& other) {
    this->check_shapes_mismatch(other);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] *= other[i][j];
        }
    }
    return *this;
}; 

Matrix Matrix::operator/(const Matrix& other) const {
    this->check_shapes_mismatch(other);
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            if (other[i][j] == 0) {
                printf("0 encountered at index (%zu, %zu)\n", i, j);
                throw std::runtime_error("Zero division in Matrix::operator/\n");
            } 
            res[i][j] /= other[i][j];
        }
    }
    return res;
}; 

Matrix Matrix::operator/=(const Matrix& other) {
    this->check_shapes_mismatch(other);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            if (other[i][j] == 0) {
                printf("0 encountered at index (%zu, %zu)\n", i, j);
                throw std::runtime_error("Zero division in Matrix::operator/=\n");
            } 
            this->vals[i][j] /= other[i][j];
        }
    }
    return *this;
}; 

Matrix Matrix::operator==(const Matrix& other) const {
    this->check_shapes_mismatch(other);
    Matrix res(this->nRows, this->nCols);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            if (this->vals[i][j] == other[i][j]) {
                res[i][j] = 1.0f;
            }
        }
    }
    return res;
};

