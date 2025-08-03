#include "utils.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

Matrix matmul(const Matrix& A, const Matrix& B) {
    int n = A.size(), k = A[0].size(), m = B[0].size();
    Matrix C(n, std::vector<double>(m, 0.0));
    
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int p = 0; p < k; ++p) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return C;
}

// Transpose: B = Aᵀ
Matrix transpose(const Matrix& A) {
    int n = A.size(), m = A[0].size();
    Matrix B(m, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[j][i] = A[i][j];
    return B;
}

// Add a column-vector B (n×1) to each column of A
Matrix add(const Matrix& A, const Matrix& B) {
    int n = A.size(), m = A[0].size();
    Matrix C = A;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] += B[i][0];
    return C;
}

// Elementwise subtraction: A − B
Matrix subtract(const Matrix& A, const Matrix& B) {
    int n = A.size(), m = A[0].size();
    Matrix C(n, std::vector<double>(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

// Multiply every element of A by scalar
Matrix scalar_multiply(const Matrix& A, double scalar) {
    int n = A.size(), m = A[0].size();
    Matrix B = A;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[i][j] *= scalar;
    return B;
}

// Sum across columns: returns n×1 column-vector
Matrix sum_columns(const Matrix& A) {
    int n = A.size(), m = A[0].size();
    Matrix B(n, std::vector<double>(1, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[i][0] += A[i][j];
    return B;
}

// Element-wise Hadamard multiplication
Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size(), m = A[0].size();
    Matrix C(n, std::vector<double>(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] = A[i][j] * B[i][j];
    return C;
}

// One-hot encoding
Matrix one_hot(const std::vector<int>& labels, int num_classes) {
    int m = labels.size();
    Matrix Y(num_classes, std::vector<double>(m, 0.0));
    for (int j = 0; j < m; ++j)
        Y[labels[j]][j] = 1.0;
    return Y;
}

// ReLU activation
Matrix relu(const Matrix& Z) {
    Matrix A = Z;
    for (auto& row : A)
        for (auto& val : row)
            val = std::max(0.0, val);
    return A;
}

// ReLU derivative (for backprop)
Matrix relu_derivative(const Matrix& Z) {
    Matrix dZ = Z;
    for (auto& row : dZ)
        for (auto& val : row)
            val = (val > 0.0 ? 1.0 : 0.0);
    return dZ;
}

// Softmax activation (column-wise)
Matrix softmax(const Matrix& Z) {
    int n = Z.size(), m = Z[0].size();
    Matrix A(n, std::vector<double>(m));
    for (int j = 0; j < m; ++j) {
        double maxv = Z[0][j];
        for (int i = 1; i < n; ++i)
            maxv = std::max(maxv, Z[i][j]);
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            A[i][j] = std::exp(Z[i][j] - maxv);
            sum += A[i][j];
        }
        for (int i = 0; i < n; ++i)
            A[i][j] /= sum;
    }
    return A;
}

// Compute mean cross-entropy loss over m samples (Y is one-hot)
double compute_loss(const Matrix& A2, const Matrix& Y) {
    int m = A2[0].size();
    double loss = 0.0;
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < (int)A2.size(); ++i) {
            if (Y[i][j] > 0.5) {
                loss -= std::log(std::max(1e-15, A2[i][j]));
            }
        }
    }
    return loss / m;
}

// Predict labels by argmax across each column
std::vector<int> predict_labels(const Matrix& A2) {
    int num_classes = A2.size(), m = A2[0].size();
    std::vector<int> preds(m);
    for (int j = 0; j < m; ++j) {
        double best = A2[0][j];
        int idx = 0;
        for (int i = 1; i < num_classes; ++i) {
            if (A2[i][j] > best) {
                best = A2[i][j];
                idx = i;
            }
        }
        preds[j] = idx;
    }
    return preds;
}

// Compute and print accuracy, precision, recall, F1 per class
void classification_report(const std::vector<int>& y_true,
                           const std::vector<int>& y_pred) {
    int m = y_true.size();
    const int C = 10;
    std::vector<int> tp(C,0), fp(C,0), fn(C,0);
    int correct = 0;

    for (int i = 0; i < m; ++i) {
        if (y_true[i] == y_pred[i]) {
            ++correct;
            ++tp[y_true[i]];
        } else {
            ++fp[y_pred[i]];
            ++fn[y_true[i]];
        }
    }

    double accuracy = double(correct) / m;
    std::cout << "Overall Accuracy: " << accuracy << "\n\n";
    std::cout << "Class | Precision | Recall | F1\n";
    std::cout << "-------------------------------\n";
    for (int c = 0; c < C; ++c) {
        double precision = (tp[c] + fp[c]) > 0
            ? double(tp[c]) / (tp[c] + fp[c]) : 0.0;
        double recall = (tp[c] + fn[c]) > 0
            ? double(tp[c]) / (tp[c] + fn[c]) : 0.0;
        double f1 = (precision + recall) > 0
            ? 2 * precision * recall / (precision + recall)
            : 0.0;
        std::cout << "  " << c << "   | "
                  << precision << "     | "
                  << recall    << "  | "
                  << f1        << "\n";
    }
}
