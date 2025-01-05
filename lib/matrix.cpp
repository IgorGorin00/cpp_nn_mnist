#include "../include/matrix.hpp"
#include <algorithm>
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
                throw std::runtime_error("Zero division in Matrix::operator/ Matrix&\n");
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
                throw std::runtime_error("Zero division in Matrix::operator/= Matrix&\n");
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

Matrix Matrix::operator+(const double x) const {
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] += x;
        }
    }
    return res;
}; 

Matrix Matrix::operator+=(const double x) {
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] += x;
        }
    }
    return *this;
}; 

Matrix Matrix::operator-(const double x) const {
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] -= x;
        }
    }
    return res;
};

Matrix Matrix::operator-=(const double x) {
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] -= x;
        }
    }
    return *this;
};

Matrix Matrix::operator*(const double x) const {
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] *= x;
        }
    }
    return res;
};

Matrix Matrix::operator*=(const double x) {
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] *= x;
        }
    }
    return *this;
};

Matrix Matrix::operator/(const double x) const {
    if (x == 0) {
        printf("0 encountered in divide!\n");
        throw std::runtime_error("Zero division in Matrix::operator/ float\n");
    }
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] /= x;
        }
    }
    return res;
};

Matrix Matrix::operator/=(const double x) {
    if (x == 0) {
        printf("0 encountered in divide!\n");
        throw std::runtime_error("Zero division in Matrix::operator/= float\n");
    }
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            this->vals[i][j] /= x;
        }
    }
    return *this;
};

Matrix Matrix::row_wise_add(const std::vector<double>& vec) const {
    if (vec.size() != this->nCols) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::row_wise_add\n");
    }
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] = this->vals[i][j] + vec[j];
        }
    }
    return res;
};

Matrix Matrix::row_wise_substract(const std::vector<double>& vec) const {
    if (vec.size() != this->nCols) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::row_wise_substract\n");
    }
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] = this->vals[i][j] - vec[j];
        }
    }
    return res;
};

Matrix Matrix::row_wise_multiply(const std::vector<double>& vec) const {
    if (vec.size() != this->nCols) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::row_wise_multiply\n");
    }
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            res[i][j] = this->vals[i][j] * vec[j];
        }
    }
    return res;
};

Matrix Matrix::row_wise_divide(const std::vector<double>& vec) const {
    if (vec.size() != this->nCols) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::row_wise_divide\n");
    }
    Matrix res(*this);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < this->nCols; j++) {
            if (vec[j] == 0) {
                printf("0 encountered in vec at index %zu\n", j);
                throw std::runtime_error("Zero division in Matrix::row_wise_divide\n");
            } 
            res[i][j] = this->vals[i][j] / vec[j];
        }
    }
    return res;
};

Matrix Matrix::col_wise_add(const std::vector<double>& vec) const {
    if (vec.size() != this->nRows) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::col_wise_add\n");
    }
    Matrix res(*this);
    for (size_t j = 0; j < this->nCols; j++) {
        for (size_t i = 0; i < this->nRows; i++) {
            res[i][j] = this->vals[i][j] + vec[i];
        }
    }
    return res;
};

Matrix Matrix::col_wise_substract(const std::vector<double>& vec) const {
    if (vec.size() != this->nRows) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::col_wise_substract\n");
    }
    Matrix res(*this);
    for (size_t j = 0; j < this->nCols; j++) {
        for (size_t i = 0; i < this->nRows; i++) {
            res[i][j] = this->vals[i][j] - vec[i];
        }
    }
    return res;
};

Matrix Matrix::col_wise_multiply(const std::vector<double>& vec) const {
    if (vec.size() != this->nRows) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::col_wise_multiply\n");
    }
    Matrix res(*this);
    for (size_t j = 0; j < this->nCols; j++) {
        for (size_t i = 0; i < this->nRows; i++) {
            res[i][j] = this->vals[i][j] * vec[i];
        }
    }
    return res;
};

Matrix Matrix::col_wise_divide(const std::vector<double>& vec) const {
    if (vec.size() != this->nRows) {
        printf("this->shape (%d, %d), vec.size() %zu\n", 
                this->nRows, this->nCols, vec.size());
        throw std::runtime_error("Shape mismatch! In Matrix::col_wise_divide\n");
    }
    Matrix res(*this);
    for (size_t j = 0; j < this->nCols; j++) {
        for (size_t i = 0; i < this->nRows; i++) {
            if (vec[j] == 0) {
                printf("0 encountered in vec at index %zu\n", i);
                throw std::runtime_error("Zero division in Matrix::col_wise_divide\n");
            } 
            res[i][j] = this->vals[i][j] / vec[i];
        }
    }
    return res;
};

Matrix Matrix::dot(const Matrix& other) const {
    if (this->nCols != other.nRows) {
        printf("this->shape (%d, %d), other.shape(%d, %d)\n",
                this->nRows, this->nCols, other.nRows, other.nCols);
        throw std::runtime_error("Shape mismatch in Matrix::dot. this->nCols should equal other.nRows!\n");
    }
    Matrix res(this->nRows, other.nCols);
    for (size_t i = 0; i < this->nRows; i++) {
        for (size_t j = 0; j < other.nCols; j++) {
            for (size_t k = 0; k < this->nCols; k++) {
                res[i][j] += this->vals[i][k] * other.vals[k][j];
            }
        }
    }
    return res;
};

Matrix Matrix::matMul(const Matrix& other) const {
    return this->dot(other);

};
