#include <algorithm>
#include <iostream>
#include <numeric>
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
                    vals[i].reserve(nc);
                    for (int j = 0; j < nc; j++) {
                        vals[i][j] = d(gen);
                    }
                }
            }
        };
    Matrix(const std::vector<double>& pre_vals)
    : nRows(pre_vals.size()), nCols(1),
    vals(pre_vals.size(), std::vector<double>(1, 0.0f)) {
        for (int i = 0; i < pre_vals.size(); i++) {
            vals[i][0] = pre_vals[i];
        }
    }

    std::vector<double>& operator[](const int idx) {
        return vals[idx];
    }

    const std::vector<double>& operator[](const int idx) const {
        return vals[idx];
    }

    Matrix operator+=(const Matrix& other) {
        if (other.nRows != nCols && other.nRows != 1) {
            this->printShapeMismatch(other);
            throw std::runtime_error("Matrix::operator+= shape mismatch!\
                    nRows should either 1 or equal to this->nCols!\n");
        }
        if (other.nCols != nCols && other.nCols != 1) {
            this->printShapeMismatch(other);
            throw std::runtime_error("Matrix::operator+= shape mismatch!\
                    Other nCols should either 1 or equal to this->nRows!\n");
        }
        if (other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] += other[j][0];
                }
            }
        }
        if (other.nRows == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] += other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] += other[i][j];
                }
            }
        }
        return *this;
    }

    Matrix transpose()  {
        Matrix res = Matrix(nCols, nRows, 0.0f);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[j][i] = vals[i][j];
            }
        }
        return res;
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

    void printShapeMismatch(const Matrix& other) const {
        printf("Shape mismatch!\nmat1 shape: (%d, %d)\nmat2 shape: (%d, %d)\n",
                nRows, nCols,
                other.nRows, other.nCols);
    }
};


Matrix matMul(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.nCols != mat2.nRows) {
        mat1.printShapeMismatch(mat2);
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

void ReLU(Matrix& mat) {
    for (int i = 0; i < mat.nRows; i++) {
        for (int j = 0; j < mat.nCols; j++) {
            if (mat[i][j] < 0) {
                mat[i][j] = 0;
            }
        }
    }
}

double MSE(const Matrix& preds, const Matrix& labels) {
    if (preds.nCols != 1 || labels.nCols != 1 || preds.nRows != labels.nRows) {
        preds.printShapeMismatch(labels);
        throw std::runtime_error("Shape mismatch in MSE!\n");
    }
    std::vector<double> diffs;
    diffs.reserve(labels.nRows);
    for (int i = 0; i < labels.nRows; i++) {
        double diff = preds[i][0] - labels[i][0];
        diffs.push_back(diff * diff);
    }
    double res = std::accumulate(diffs.begin(), diffs.end(), 0.0f);
    return res;
}

struct Linear {
public:
    const int inDim, outDim;
    Matrix weight;
    Matrix bias;
    Matrix weightGrad;
    Matrix biasGrad;
    Matrix lastOut;

    Linear(const int in_dim, const int out_dim)
    : inDim(in_dim), outDim(out_dim),
    weight(Matrix(out_dim, in_dim, 0.0f, 2.0f / (in_dim + out_dim))),
    bias(Matrix(1, out_dim, 0.0f)),
    weightGrad(out_dim, in_dim, 0.0f),
    biasGrad(1, out_dim, 0.0f),
    lastOut(1, 1)
    {};

    Matrix forward(const Matrix& vals) {
        Matrix res = matMul(vals, weight.transpose());
        res += bias;
        lastOut = res;
        return res;
    } 
};


struct Network {
public:
    Linear layer1;
    Linear layer2;
    Linear layer3;

    Network(const int in_dim, const int hidden_dim, const int out_dim)
    : layer1(in_dim, hidden_dim), 
    layer2(hidden_dim, hidden_dim),
    layer3(hidden_dim, out_dim)
    {};

    Matrix forward(const Matrix& vals) {
        Matrix res = layer1.forward(vals);
        ReLU(res);
        res = layer2.forward(res);
        ReLU(res);
        res = layer3.forward(res);
        return res;
    }
};

std::vector<double> linspace(const double low, const double high,
        const int n_samples) {
    if (n_samples == 0) {
        throw std::runtime_error("cant divide by 0 in linspace!\n");
    }
    std::vector<double> res(n_samples);
    res.reserve(n_samples);
    const double step = (high - low) / n_samples;
    for (int i = 0; i < n_samples; i++) {
        res[i] = low + step * i;
    }
    return res;
}

Matrix getLabels(const double low, const double high, const int n_samples,
        const float mean, const float scale) {
    std::vector<double> pre_res = linspace(low, high, n_samples);
    Matrix mat(n_samples, 1, mean, scale);

    std::random_device rd{};
    std::mt19937 gen(rd());
    std::normal_distribution<double> d{mean, scale};
    for (int i = 0; i < n_samples; i++) {
        mat[i][0] = std::sin(pre_res[i]) * d(gen);
    }
    return mat;
}


int main() {
    const double low = -10.0f;
    const double high = 10.0f;
    const int n_samples = 1000;
    const double mean = 1.0f;
    const double scale = 0.5f;
    std::vector<double> pre_vals = linspace(low, high, n_samples);
    Matrix vals = Matrix(pre_vals);
    Matrix labels = getLabels(low, high, n_samples, mean, scale);

    const int in_dim = vals.nCols;
    const int hidden_dim = 10;
    const int out_dim = labels.nCols;
    Network net = Network(in_dim, hidden_dim, out_dim);
    

    Matrix out = net.forward(vals);

    double loss = MSE(out, labels);
    printf("%f\n", loss);


    return 0;
}
