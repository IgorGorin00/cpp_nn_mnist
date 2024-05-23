#include <fstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <random>
#include <cassert>

struct Matrix {
public:
    int nRows, nCols;
    std::vector<std::vector<double>> vals;
    
    Matrix(const int nr, const int nc, 
           const double mean = 0.0f, const double scale = 0.0f) 
        : nRows(nr), nCols(nc), vals(nr, std::vector<double>(nc, mean)) {
            if (scale != 0.0f) {
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
        for (size_t i = 0; i < pre_vals.size(); i++) {
            vals[i][0] = pre_vals[i];
        }
    }

    Matrix(const std::vector<std::vector<double>>& pre_vals)
    : nRows(pre_vals.size()), nCols(pre_vals[0].size()),
    vals(pre_vals) {}

    std::vector<double>& operator[](const int idx) {
        return vals[idx];
    }

    const std::vector<double>& operator[](const int idx) const {
        return vals[idx];
    }

    Matrix operator+(const float x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] += x;
            }
        }
        return res;
    }

    Matrix operator-(const float x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] -= x;
            }
        }
        return res;
    }

    Matrix operator*(const float x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] *= x;
            }
        }
        return res;
    }

    Matrix operator/(const float x) const {
        Matrix res(*this);
        if (x == 0.0f) {
            throw std::runtime_error("cant divide by 0 in matrix overloading!\n");
        }
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] /= x;
            }
        }
        return res;
    }

    Matrix operator+=(const float x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] += x;
            }
        }
        return *this;
    }

    Matrix operator-=(const float x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] -= x;
            }
        }
        return *this;
    }

    Matrix operator*=(const float x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] *= x;
            }
        }
        return *this;
    }

    Matrix operator/=(const float x) {
        if (x == 0.0f) {
            throw std::runtime_error("cant divide by 0 in matrix overloading!\n");
        }
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] /= x;
            }
        }
        return *this;
    }

    void checkShapeMismatch(const Matrix& other) const {
        if (other.nRows != 1 && other.nRows != nRows) {
            this->printShapeMismatch(other);
            throw std::runtime_error("Matrix::operator+-*= shape mismatch!\n"
                         "nRows should either 1 or equal to this->nRows!\n");
        }
        if (other.nCols != nCols && other.nCols != 1) {
            this->printShapeMismatch(other);
            throw std::runtime_error("Matrix::operator+-*= shape mismatch!\n"
                    "Other nCols should either 1 or equal to this->nRows!\n");
        }
    }

    Matrix operator+(const Matrix& other) const {
        this->checkShapeMismatch(other);
        Matrix res(*this);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] += other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] += other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] += other[i][j];
                }
            }
        }
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        this->checkShapeMismatch(other);
        Matrix res(*this);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] -= other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] -= other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] -= other[i][j];
                }
            }
        }
        return res;
    }

    Matrix operator*(const Matrix& other) const {
        this->checkShapeMismatch(other);
        Matrix res(*this);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] *= other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] *= other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    res[i][j] *= other[i][j];
                }
            }
        }
        return res;
    }

    Matrix operator+=(const Matrix& other) {
        this->checkShapeMismatch(other);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] += other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
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

    Matrix operator-=(const Matrix& other) {
        this->checkShapeMismatch(other);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] -= other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] -= other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] -= other[i][j];
                }
            }
        }
        return *this;
    }

    Matrix operator*=(const Matrix& other) {
        this->checkShapeMismatch(other);
        if (other.nRows == nRows && other.nCols == 1) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] *= other[i][0];
                }
            }
        }
        if (other.nRows == 1 && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] *= other[0][j];
                }
            }
        }
        if (other.nRows == nRows && other.nCols == nCols) {
            for (int i = 0; i < nRows; i++) {
                for (int j = 0; j < nCols; j++) {
                    vals[i][j] *= other[i][j];
                }
            }
        }
        return *this;
    }

    Matrix colWiseSum() const {
        Matrix res(this->nRows, 1, 0.0f);
        for (int j = 0; j < nCols; j++) {
            for (int i = 0; i < nRows; i++) {
                res[i][0] += vals[i][j];
            }
        }
        return res;
    }

    Matrix rowWiseSum() const {
        Matrix res(1, this->nCols, 0.0f);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[0][j] += vals[i][j];
            }
        }
        return res;
    }

    Matrix rowWiseMean() const {
        if (nRows == 0) {
            throw std::runtime_error("cant do rowWise mean when nRowos is 0!\n");
        }
        Matrix res(nRows, 1, 0.0f);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][0] += vals[i][j];
            }
        }
        for (int i = 0; i < nRows; i++) {
            res[i][0] /= nRows;
        }
        return res;
    }

    Matrix colWiseMean() const {
        if (nCols == 0) {
            throw std::runtime_error("cant do colWise mean when nRowos is 0!\n");
        }
        Matrix res(1, nCols, 0.0f);
        for (int j = 0; j < nCols; j++) {
            for (int i = 0; i < nRows; i++) {
                res[0][j] += vals[i][j];
            }
        }
        for (int j = 0; j < nCols; j++) {
            res[0][j] /= nCols;
        }
        return res;
    }
    
    Matrix mean(const int axis = 0) const {
        if (axis == 0) {
            return this->rowWiseMean();
        } else if (axis == 1) {
            return this->colWiseMean();
        } else {
            throw std::runtime_error("axis should be either 0 or 1 in mean!\n");
        }

        
    }

    double sum() const {
        double res = 0.0f;
        for (const std::vector<double>& row : vals) {
            for (const double n : row) {
                res += n;
            }
        }
        return res;
    }

    Matrix transpose()  {
        Matrix res(nCols, nRows, 0.0f);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[j][i] = vals[i][j];
            }
        }
        return res;
    }

    Matrix T() {
        return this->transpose();
    }

    Matrix rowSlice(const int idx_start, const int idx_end) const {
        int res_n_rows = idx_end - idx_start;
        if (res_n_rows <= 0) {
            throw std::runtime_error("cant slice <= 0 rows");
        }
        Matrix res(res_n_rows, this->nCols);
        for (int i = 0; i < res_n_rows; i++) {
            res[i] = this->vals[i];
        }
        return res;
    }

    void print() const {
        printf("{\n");
        for (const std::vector<double> &row : vals) {
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
                this->nRows, this->nCols,
                other.nRows, other.nCols);
    }
};

template <typename T>
void print_vector(const std::vector<T> vec) {
    printf("{");
    for (const T e : vec) {
        std::cout << e << " ";
    }
    printf("}\n");
}


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

Matrix ReLU(const Matrix& mat) {
    Matrix res(mat.nRows, mat.nCols, 0.0f);
    for (int i = 0; i < mat.nRows; i++) {
        for (int j = 0; j < mat.nCols; j++) {
            if (mat[i][j] < 0) {
                res[i][j] = 0;
            } else {
                res[i][j] = mat[i][j];
            }
        }
    }
    return res;
}

Matrix dReLU(const Matrix& mat) {
    Matrix res(mat.nRows, mat.nCols, 0.0f);
    for (int i = 0; i < mat.nRows; i++) {
        for (int j = 0; j < mat.nCols; j++) {
            if (mat[i][j] > 0) {
                res[i][j] = 1.0f;
            }
        }
    }
    return res;
}

double exp_sum(const double sum, const double num) {
    return sum + std::exp(num);
}


Matrix softmax(const Matrix& mat) {
    Matrix res(mat.nRows, mat.nCols, 0.0f);
    for (int i = 0; i < mat.nRows; i++) {
        double max_val = *std::max_element(mat[i].begin(), mat[i].end());
        
        double row_exp_sum = 0.0;
        for (int j = 0; j < mat.nCols; j++) {
            row_exp_sum += std::exp(mat[i][j] - max_val);
        }
        
        for (int j = 0; j < mat.nCols; j++) {
            double softmax_val = std::exp(mat[i][j] - max_val) / row_exp_sum;
            res[i][j] = softmax_val;
        }
    }
    return res;
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
    return res / labels.nRows;
}

double crossEntropyLoss(const Matrix& preds, const Matrix& labels) {
    double res = 0.0f;
    preds.checkShapeMismatch(labels);
    for (int i = 0; i < preds.nRows; i++) {
        double entry_diff = 0.0f;
        for (int j = 0; j < preds.nCols; j++) {
            entry_diff += preds[i][j] * std::log(labels[i][j] + 1e-16);
        }
        res -= entry_diff;
    }
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

    void zero_grad() {
        weightGrad = Matrix(outDim, inDim, 0.0f);
        biasGrad = Matrix(1, outDim, 0.0f);
    }
};


double accuracy(const Matrix& preds, const Matrix& labels) {
    preds.checkShapeMismatch(labels);
    int n_correct = 0;
    for (int i = 0; i < preds.nRows; i++) {
        int max_idx_pred = -1;
        int max_idx_label = -1;
        double max_pred = 0.0f;
        double max_label = 0.0f;
        for (int j = 0; j < preds.nCols; j++) {
            if (preds[i][j] > max_pred) {
                max_pred = preds[i][j]; 
                max_idx_pred = j;
            }
            if (labels[i][j] > max_label) {
                max_label = labels[i][j]; 
                max_idx_label = j;
            }
        }
        if (max_idx_pred == -1 || max_idx_label == -1) {
            print_vector(preds[i]);
            print_vector(labels[i]);
            printf("max_idx_pred = %d, max_idx_label = %d\n",
                    max_idx_pred, max_idx_label);
            printf("hello world\n");
            throw std::runtime_error("not found max idx for label or pred!\n");
        }
        if (max_idx_pred == max_idx_label) {
            n_correct++;
        }
    }
    double accuracy = static_cast<double>(n_correct) / 
                      static_cast<double>(preds.nRows);
    return accuracy;
}


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
        Matrix res = this->layer1.forward(vals);
        res = ReLU(res);
        res = this->layer2.forward(res);
        res = ReLU(res);
        res = this->layer3.forward(res);
        res = softmax(res);
        this->layer3.lastOut = res;
        return res;
    }
    
    void backward(const Matrix& vals, const Matrix& labels) {
        Matrix loss_grad = layer3.lastOut - labels;

        Matrix delta_layer3 = loss_grad;

        this->layer3.weightGrad = matMul(
                delta_layer3.T(), ReLU(this->layer2.lastOut));
        this->layer3.biasGrad = delta_layer3.rowWiseSum();
        Matrix delta_layer2 = matMul(delta_layer3, this->layer3.weight) *\
                              dReLU(this->layer2.lastOut);

        this->layer2.weightGrad = matMul(
                delta_layer2.T(), ReLU(this->layer1.lastOut));
        this->layer2.biasGrad = delta_layer2.rowWiseSum();
        Matrix delta_layer1 = matMul(delta_layer2, this->layer2.weight) *\
                              dReLU(this->layer1.lastOut);

        this->layer1.weightGrad = matMul(delta_layer1.T(), vals);
        this->layer1.biasGrad = delta_layer1.rowWiseSum();
    }
    
    void updateWeigths(const double lr) {
        this->layer1.weightGrad *= lr;
        this->layer1.biasGrad *= lr;
        this->layer1.weight -= this->layer1.weightGrad;
        this->layer1.bias -= this->layer1.biasGrad;

        this->layer2.weightGrad *= lr;
        this->layer2.biasGrad *= lr;
        this->layer2.weight -= this->layer2.weightGrad;
        this->layer2.bias -= this->layer2.biasGrad;

        this->layer3.weightGrad *= lr;
        this->layer3.biasGrad *= lr;
        this->layer3.weight -= this->layer3.weightGrad;
        this->layer3.bias -= this->layer3.biasGrad;
    }

    void zeroGrad() {
        for (Linear& layer : std::vector<Linear>{layer1, layer2, layer3}) {
            layer.zero_grad();
        }
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




std::vector<std::vector<double>> readMNISTImages(
        const std::string& filename, int n_imgs) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    int magic_number, n_imgs_, n_rows, n_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&n_imgs_), sizeof(n_imgs_));
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    magic_number = __builtin_bswap32(magic_number);
    n_imgs_ = __builtin_bswap32(n_imgs_);
    n_rows = __builtin_bswap32(n_rows);
    n_cols = __builtin_bswap32(n_cols);

    if (magic_number != 2051) {
        std::cerr << "Invalid magic number, expected 2051, got "
            << magic_number << std::endl;
        return {};
    }

    if (n_imgs > n_imgs_) {
        std::cerr << "Requested more images than available in the dataset"
            << std::endl;
        return {};
    }

    std::vector<std::vector<double>> images(
            n_imgs, std::vector<double>(n_rows * n_cols));

    for (int i = 0; i < n_imgs; ++i) {
        for (int j = 0; j < n_rows * n_cols; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0; // Normalize pixel values to [0, 1]
            if (std::isnan(images[i][j])) {
                printf("%c / 255.0 results in -nan\n", pixel);
                throw std::runtime_error("nan in reading images!\n");
            }
        }
    }

    return images;
}

std::vector<int> readMNISTLabels(const std::string& filename, int n_labels) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    int magic_number, n_labels_;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&n_labels_), sizeof(n_labels_));

    magic_number = __builtin_bswap32(magic_number);
    n_labels_ = __builtin_bswap32(n_labels_);

    if (magic_number != 2049) {
        std::cerr << "Invalid magic number, expected 2049, got "
            << magic_number << std::endl;
        return {};
    }

    if (n_labels > n_labels_) {
        std::cerr << "Requested more labels than available in the dataset"
            << std::endl;
        return {};
    }

    std::vector<int> labels(n_labels);
    for (int i = 0; i < n_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }

    return labels;
}


Matrix encodeLabels(const std::vector<int>& vec, const int n_classes) {
    Matrix res(vec.size(), n_classes);
    for (size_t i = 0; i < vec.size(); i++) {
        for (int j = 0; j < n_classes; j++) {
            if (vec[i] == j) {
                res[i][j] = 1.0f;
            }
        }
    }
    return res;
}

void train(const int n_epochs, const int batch_size, 
           const Matrix& images, const Matrix& labels, 
           Network& net, const double lr) {
    for (int i = 0; i < n_epochs; i++) {
        for (int b_start = 0; b_start < images.nRows; b_start += batch_size) {
            int b_end = b_start + batch_size;
            Matrix b_images = images.rowSlice(b_start, b_end);
            Matrix b_labels = labels.rowSlice(b_start, b_end);
            Matrix out = net.forward(b_images);
            net.backward(b_images, b_labels);
            net.updateWeigths(lr);
        }
        double acc = 0.0f;
        double loss = 0.0f;
        for (int b_start = 0; b_start < images.nRows; b_start += batch_size) {
            int b_end = b_start + batch_size;
            Matrix b_images = images.rowSlice(b_start, b_end);
            Matrix b_labels = labels.rowSlice(b_start, b_end);
            Matrix out = net.forward(b_images);
            out = softmax(out);
            acc += accuracy(out, b_labels);
            loss += crossEntropyLoss(out, b_labels);
        }
        double epoch_acc = acc / images.nRows;
        double epoch_loss = loss / images.nRows;
        printf("Epoch = %d\tAccuracy = %f\tLoss = %f\n", i, epoch_acc, epoch_loss);

    }
}


int main() {
    const std::string images_file = "../public/train_images.ubyte";
    const std::string labels_file = "../public/train_labels.ubyte";
    const int num_images = 60000; // Number of images in the dataset

    // Read images and labels
    std::vector<std::vector<double>> images = readMNISTImages(images_file, num_images);
    std::vector<int> labels = readMNISTLabels(labels_file, num_images);

    Matrix images_mat = Matrix(images);
    Matrix labels_mat = encodeLabels(labels, 10);
    
    Network net(images_mat.nCols, 32, labels_mat.nCols);

    const int n_epochs = 100;
    const int batch_size = 64;
    const double lr = 3e-3;
    train(n_epochs, batch_size, images_mat, labels_mat, net, lr);
    return 0;
}
