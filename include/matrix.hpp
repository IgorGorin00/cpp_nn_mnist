#include <vector>

struct Matrix {
public:
    std::vector<std::vector<double>> vals;
    int nRows, nCols;

    Matrix(const int n_rows_, const int n_cols_, const double mean = 0.0f, const double scale = 0.0f);
    Matrix(std::vector<std::vector<double>>& pre_vals);
    void print() const;
    std::vector<double>& operator[](const int idx); 
    const std::vector<double>& operator[](const int idx) const; 
    void check_shapes_mismatch(const Matrix& other) const;
    Matrix operator+(const Matrix& other) const; 
    Matrix operator+=(const Matrix& other); 
    Matrix operator-(const Matrix& other) const; 
    Matrix operator-=(const Matrix& other); 
    Matrix operator*(const Matrix& other) const; 
    Matrix operator*=(const Matrix& other); 
    Matrix operator/(const Matrix& other) const; 
    Matrix operator/=(const Matrix& other); 
    Matrix operator==(const Matrix& other) const; 
};
