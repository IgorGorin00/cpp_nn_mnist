#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <random>




struct Matrix {
public:
    int nRows, nCols;
    std::vector<std::vector<double>> vals;
    
    Matrix(const int nr, const int nc, 
           const float mean = 0.0, const float scale = 0.0) 
        : nRows(nr), nCols(nc), vals(nr, std::vector<double>(nc, mean)) {
            if (scale != 0.0) {
                std::random_device rd{};
                std::mt19937 gen(rd());
                std::normal_distribution<double> d{mean, scale};
                for (int i = 0; i < nr; i++) {
                    for (int j = 0; j < nc; j++) {
                        vals[i][j] = d(gen);
                    }
                }
            }
        };

    std::vector<double> operator[](const int idx) {
        return vals[idx];
    }

    const std::vector<double> operator[](const int idx) const {
        return vals[idx];
    }


    void print() const {
        printf("{\n");
        for (const std::vector<double> row : vals) {
            printf("{ ");
            for (const double n : row) {
                printf("%f ", n);
            }
            printf("},\n");
        }
        printf("}\n");
    }
};


Matrix matMul(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.nCols != mat2.nRows) {
        printf("Shape mismatch!\
                mat1 shape: (%d, %d)\
                mat2 shape: (%d, %d)\n",
                mat1.nRows, mat1.nCols,
                mat2.nRows, mat2.nCols);
        throw std::runtime_error("matMul shape mismatch!");
    }
    Matrix res = Matrix(mat1.nRows, mat2.nCols);
    for (int i = 0; i < mat1.nRows; i++) {
        for (int j = 0; j < mat2.nCols; j++) {
            for (int k = 0; k < mat2.nRows; k++) {
                res[i][j] += mat1[i][k] * mat2[k][j]; 
            }
        }
    }
    return res;
}



int main() {
    Matrix mat = Matrix(3, 3);
    mat.print();

    return 0;
}
