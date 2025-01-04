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

    Matrix operator+(const double x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] += x;
            }
        }
        return res;
    }

    Matrix operator-(const double x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] -= x;
            }
        }
        return res;
    }

    Matrix operator*(const double x) const {
        Matrix res(*this);
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                res[i][j] *= x;
            }
        }
        return res;
    }

    Matrix operator/(const double x) const {
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

    Matrix operator+=(const double x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] += x;
            }
        }
        return *this;
    }

    Matrix operator-=(const double x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] -= x;
            }
        }
        return *this;
    }

    Matrix operator*=(const double x) {
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                vals[i][j] *= x;
            }
        }
        return *this;
    }

    Matrix operator/=(const double x) {
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

    Matrix transpose() {
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

Matrix one_hot_encoding(const std::vector<int>& vec, const int n_classes) {
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


std::vector<Matrix> init_params() {
    Matrix W1(784, 10, 0.0f, 0.5f);
    Matrix b1(1, 10, 0.0f, 0.5f);
    Matrix W2(10, 10, 0.0f, 0.5f);
    Matrix b2 (1, 10, 0.0f, 0.5f);
    return std::vector<Matrix>{W1, b1, W2, b2};
}


std::vector<Matrix> forward_prop(Matrix& W1, Matrix& b1, Matrix& W2, Matrix& b2, Matrix& X) {
    Matrix Z1 = matMul(W1, X);
    Z1 += b1;
    Matrix A1 = ReLU(Z1);
    Matrix Z2 = matMul(W2, A1);
    Z2 += b2;
    Matrix A2 = softmax(Z2);
    return std::vector<Matrix>{Z1, A1, Z2, A2};
}


std::vector<Matrix> backward_prob(
        Matrix& Z1,
        Matrix& A1,
        Matrix& Z2,
        Matrix& A2,
        Matrix& W1,
        Matrix& W2,
        Matrix& X,
        Matrix& Y) {
    int m = X.nRows;
    Matrix dZ2 = A2 - Y;
    
    Matrix dW2 = matMul(dZ2, A1.T());
    dW2 *= 1 / m;

    Matrix db2 = dZ2.rowWiseSum();
    db2 *= 1 / m;

    Matrix dZ1 = matMul(W2.T(), dZ2) * dReLU(Z1);

    Matrix dW1 = matMul(dZ1, X.T());
    dW1 *= 1 / m;

    Matrix db1 = dZ1.rowWiseSum();
    db1 *= 1 / m;

    return std::vector<Matrix>{dW1, db1, dW2, db1};
}


void update_params(
        Matrix& W1,
        Matrix& b1,
        Matrix& W2,
        Matrix& b2,
        Matrix& dW1,
        Matrix& db1,
        Matrix& dW2,
        Matrix& db2,
        double alpha) {
    W1 -= dW1 * alpha;
    b1 -= db1 * alpha;
    W2 -= dW2 * alpha;
    b2 -= db2 * alpha;
}


std::vector<int> get_predictions(const Matrix& A2) {
    std::vector<int> res(A2.nRows, 0);
    for (int i = 0; i < A2.nRows; i++) {
        int idx_max_elem = std::max_element(A2[i].begin(), A2[i].end()) - A2[i].begin();
        res[i] = idx_max_elem;
    }
    return res;
}

double get_accuracy(
        const std::vector<int>& predictions,
        const std::vector<int>& Y) {
    int n_correct = 0;
    if (predictions.size() != Y.size()) {
        printf("Prds size = %zu\n Labels size = %zu\n",
                predictions.size(), Y.size());
        throw std::runtime_error("Preds and labels size mismatch!\n");
    }
    for (int i = 0; i < Y.size(); i++) {
        if (predictions[i] == Y[i]) {
            n_correct++;
        }
    }
    double res = static_cast<double>(n_correct) / static_cast<double>(Y.size());
    return res;
}

int main() {

    const std::string images_file = "../public/train_images.ubyte";
    const std::string labels_file = "../public/train_labels.ubyte";
    const int num_images = 10; //60000; // Number of images in the dataset

    // Read images and labels
    std::vector<std::vector<double>> images = readMNISTImages(images_file, num_images);
    std::vector<int> labels = readMNISTLabels(labels_file, num_images);

    Matrix images_mat = Matrix(images);
    Matrix labels_mat = one_hot_encoding(labels, 10);

    printf("Images_mat: (%d, %d)\n", images_mat.nRows, images_mat.nCols);

    std::vector<Matrix> params = init_params();
    std::cout << params.size() << "\n";
    Matrix W1 = params[0];
    Matrix b1 = params[1];
    Matrix W2 = params[2];
    Matrix b2 = params[3];

    std::vector<Matrix> forward_res = forward_prop(W1, b1, W2, b2, images_mat);


    return 0;
}


