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

    Matrix operator+(const double x) const; 
    Matrix operator+=(const double x); 
    Matrix operator-(const double x) const; 
    Matrix operator-=(const double x); 
    Matrix operator*(const double x) const; 
    Matrix operator*=(const double x); 
    Matrix operator/(const double x) const; 
    Matrix operator/=(const double x); 

    Matrix row_wise_add(const std::vector<double>& vec) const;
    Matrix row_wise_substract(const std::vector<double>& vec) const;
    Matrix row_wise_multiply(const std::vector<double>& vec) const;
    Matrix row_wise_divide(const std::vector<double>& vec) const;
    Matrix col_wise_add(const std::vector<double>& vec) const;
    Matrix col_wise_substract(const std::vector<double>& vec) const;
    Matrix col_wise_multiply(const std::vector<double>& vec) const;
    Matrix col_wise_divide(const std::vector<double>& vec) const;

    Matrix dot(const Matrix& other) const;
    Matrix matMul(const Matrix& other) const;
};
