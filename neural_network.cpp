#include "neural_network.hpp"
#include <random>
#include <cmath>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double lr)
    : learning_rate(lr)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> dist1(0.0, sqrt(2.0 / input_size));
    std::normal_distribution<> dist2(0.0, sqrt(2.0 / hidden_size));

    W1 = Matrix(hidden_size, std::vector<double>(input_size));
    b1 = Matrix(hidden_size, std::vector<double>(1, 0.0));
    W2 = Matrix(output_size, std::vector<double>(hidden_size));
    b2 = Matrix(output_size, std::vector<double>(1, 0.0));

    for (auto& row : W1)
        for (auto& v : row)
            v = dist1(gen);

    for (auto& row : W2)
        for (auto& v : row)
            v = dist2(gen);
}

void NeuralNetwork::forward(const Matrix& X, Matrix& Z1, Matrix& A1, Matrix& Z2, Matrix& A2) const {
    Z1 = add(matmul(W1, X), b1);
    A1 = relu(Z1);
    Z2 = add(matmul(W2, A1), b2);
    A2 = softmax(Z2);
}

void NeuralNetwork::backward(const Matrix& X, const Matrix& Y,
                             const Matrix& Z1, const Matrix& A1, const Matrix& A2) {
    int m = X[0].size();

    Matrix dZ2 = subtract(A2, Y);
    Matrix dW2 = scalar_multiply(matmul(dZ2, transpose(A1)), 1.0 / m);
    Matrix db2 = scalar_multiply(sum_columns(dZ2), 1.0 / m);

    Matrix dZ1_pre = matmul(transpose(W2), dZ2);
    Matrix dZ1 = multiply(dZ1_pre, relu_derivative(Z1));

    Matrix dW1 = scalar_multiply(matmul(dZ1, transpose(X)), 1.0 / m);
    Matrix db1 = scalar_multiply(sum_columns(dZ1), 1.0 / m);

    double clip_value = 5.0;
    auto clip_matrix = [clip_value](Matrix& mat) {
        for (auto& row : mat) {
            for (auto& val : row) {
                val = std::max(-clip_value, std::min(clip_value, val));
            }
        }
    };
    
    clip_matrix(dW1);
    clip_matrix(dW2);
    clip_matrix(db1);
    clip_matrix(db2);

    W2 = subtract(W2, scalar_multiply(dW2, learning_rate));
    b2 = subtract(b2, scalar_multiply(db2, learning_rate));
    W1 = subtract(W1, scalar_multiply(dW1, learning_rate));
    b1 = subtract(b1, scalar_multiply(db1, learning_rate));
}
