#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "utils.hpp"

class NeuralNetwork {
public:
    Matrix W1, b1;
    Matrix W2, b2;
    double learning_rate;

    NeuralNetwork(int input_size, int hidden_size, int output_size, double lr);

    void forward(const Matrix& X, Matrix& Z1, Matrix& A1, Matrix& Z2, Matrix& A2) const;
    void backward(const Matrix& X, const Matrix& Y,
                  const Matrix& Z1, const Matrix& A1, const Matrix& A2);
};

#endif // NEURAL_NETWORK_HPP
