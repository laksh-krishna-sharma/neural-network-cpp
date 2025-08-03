#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include <algorithm>

using Matrix = std::vector<std::vector<double>>;

// Basic matrix operations
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix add(const Matrix& A, const Matrix& B);            // A + column-vector B
Matrix subtract(const Matrix& A, const Matrix& B);       // elementwise A − B
Matrix scalar_multiply(const Matrix& A, double scalar);
Matrix sum_columns(const Matrix& A);
Matrix multiply(const Matrix& A, const Matrix& B);       // elementwise Hadamard
Matrix one_hot(const std::vector<int>& labels, int num_classes);

// Activation functions
Matrix relu(const Matrix& Z);
Matrix relu_derivative(const Matrix& Z);
Matrix softmax(const Matrix& Z);

// Loss
// Compute mean cross-entropy loss over m samples (Y is one-hot)
double compute_loss(const Matrix& A2, const Matrix& Y);

// Prediction & metrics
std::vector<int> predict_labels(const Matrix& A2);
void classification_report(const std::vector<int>& y_true,
                           const std::vector<int>& y_pred);

#endif // UTILS_HPP
