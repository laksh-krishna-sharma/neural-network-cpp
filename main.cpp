#include "neural_network.hpp"
#include "csv.hpp"
#include "utils.hpp"

#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <string>

int main() {
    std::cout << "Loading dataset..." << std::endl;
    Matrix X;
    std::vector<int> Y_labels;
    load_csv("../digit_recognizer.csv", X, Y_labels);
    std::cout << "Loaded " << Y_labels.size() << " samples." << std::endl;

    int m = Y_labels.size();
    int train_sz = static_cast<int>(m * 0.8);
    int test_sz  = m - train_sz;

    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng{std::random_device{}()};
    std::shuffle(idx.begin(), idx.end(), rng);

    Matrix X_train(784, std::vector<double>(train_sz));
    Matrix X_test (784, std::vector<double>(test_sz));
    std::vector<int> Y_train_lbl(train_sz), Y_test_lbl(test_sz);

    for (int i = 0; i < train_sz; ++i) {
        int j = idx[i];
        Y_train_lbl[i] = Y_labels[j];
        for (int p = 0; p < 784; ++p) X_train[p][i] = X[p][j];
    }
    for (int i = train_sz; i < m; ++i) {
        int j = idx[i];
        Y_test_lbl[i - train_sz] = Y_labels[j];
        for (int p = 0; p < 784; ++p) X_test[p][i - train_sz] = X[p][j];
    }

    for (int i = 0; i < 784; ++i) {
        for (int j = 0; j < train_sz; ++j) {
            X_train[i][j] /= 255.0;
        }
        for (int j = 0; j < test_sz; ++j) {
            X_test[i][j] /= 255.0;
        }
    }

    Matrix Y_train = one_hot(Y_train_lbl, 10);

    int input_size = 784, hidden_units = 256, output_size = 10;
    double learning_rate = 0.01;
    NeuralNetwork nn(input_size, hidden_units, output_size, learning_rate);

    int batch_size = 64, epochs = 15, num_batches = train_sz / batch_size;

    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.begin() + train_sz, rng);

        for (int b = 0; b < num_batches; ++b) {
            Matrix Xb(input_size, std::vector<double>(batch_size));
            Matrix Yb(output_size, std::vector<double>(batch_size));
            for (int k = 0; k < batch_size; ++k) {
                int j = idx[b * batch_size + k];
                for (int p = 0; p < input_size; ++p) Xb[p][k] = X_train[p][j];
                for (int c = 0; c < output_size; ++c) Yb[c][k] = Y_train[c][j];
            }
            Matrix Z1, A1, Z2, A2;
            nn.forward(Xb, Z1, A1, Z2, A2);
            nn.backward(Xb, Yb, Z1, A1, A2);
        }

        Matrix Z1t, A1t, Z2t, A2t;
        nn.forward(X_train, Z1t, A1t, Z2t, A2t);

        double train_loss = compute_loss(A2t, Y_train);
        auto train_pred = predict_labels(A2t);

        int correct = 0;
        for (int i = 0; i < train_sz; ++i)
            if (train_pred[i] == Y_train_lbl[i]) ++correct;
        double train_acc = double(correct) / train_sz;

        std::cout << "Epoch " << (e + 1) << "/" << epochs
                  << " — loss: " << train_loss
                  << " — acc:  " << train_acc << "\n";

        if ((e + 1) % 10 == 0)
            nn.learning_rate *= 0.5;
    }

    Matrix Z1e, A1e, Z2e, A2e;
    nn.forward(X_test, Z1e, A1e, Z2e, A2e);

    auto Y_pred = predict_labels(A2e);
    classification_report(Y_test_lbl, Y_pred);
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "INTERACTIVE PREDICTION MODE" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    std::cout << "Choose mode:" << std::endl;
    std::cout << "1. Test existing samples (enter sample index 0-" << (test_sz-1) << ")" << std::endl;
    std::cout << "2. Draw your own digit (enter 'draw')" << std::endl;
    std::cout << "3. Exit (enter -1)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Sample preview (first 10 test samples):" << std::endl;
    for (int i = 0; i < std::min(10, test_sz); ++i) {
        std::cout << "Sample " << i << ": digit " << Y_test_lbl[i] << std::endl;
    }
    std::cout << std::endl;
    
    std::string input;
    while (true) {
        std::cout << "Enter your choice: ";
        std::cin >> input;
        
        if (input == "-1") break;
        
        if (input == "draw") {
            std::cout << "\n=== DRAW YOUR DIGIT ===" << std::endl;
            std::cout << "Draw a 28x28 digit using '#' for dark pixels and '.' for light pixels" << std::endl;
            std::cout << "Enter 28 lines of 28 characters each:" << std::endl;
            
            Matrix drawn_digit(784, std::vector<double>(1, 0.0));
            std::string line;
            std::getline(std::cin, line); // consume newline
            
            for (int row = 0; row < 28; ++row) {
                std::getline(std::cin, line);
                if (line.length() < 28) {
                    line.resize(28, '.');
                }
                for (int col = 0; col < 28; ++col) {
                    char pixel = (col < line.length()) ? line[col] : '.';
                    double value;
                    if (pixel == '#' || pixel == '*' || pixel == '@') {
                        value = 255.0; // dark pixel
                    } else if (pixel == '+' || pixel == 'o') {
                        value = 128.0; // medium pixel
                    } else {
                        value = 0.0; // light pixel
                    }
                    drawn_digit[row * 28 + col][0] = value / 255.0; // normalize
                }
            }
            
            Matrix Z1d, A1d, Z2d, A2d;
            nn.forward(drawn_digit, Z1d, A1d, Z2d, A2d);
            auto prediction = predict_labels(A2d)[0];
            
            std::cout << "\n--- YOUR DRAWN DIGIT ---" << std::endl;
            std::cout << "Predicted digit: " << prediction << std::endl;
            std::cout << "Confidence scores:" << std::endl;
            
            for (int i = 0; i < 10; ++i) {
                double confidence = A2d[i][0] * 100;
                std::cout << "  Digit " << i << ": " << std::fixed << std::setprecision(2) 
                          << confidence << "%";
                if (i == prediction) std::cout << " ← PREDICTED";
                std::cout << std::endl;
            }
            std::cout << std::endl;
            
        } else {
            try {
                int sample_idx = std::stoi(input);
                if (sample_idx < 0 || sample_idx >= test_sz) {
                    std::cout << "Invalid index! Please enter 0-" << (test_sz-1) << ", 'draw', or -1 to exit." << std::endl;
                    continue;
                }
                
                Matrix single_sample(784, std::vector<double>(1));
                for (int i = 0; i < 784; ++i) {
                    single_sample[i][0] = X_test[i][sample_idx];
                }
                
                Matrix Z1s, A1s, Z2s, A2s;
                nn.forward(single_sample, Z1s, A1s, Z2s, A2s);
                auto prediction = predict_labels(A2s)[0];
                std::cout << "\n--- Sample #" << sample_idx << " ---" << std::endl;
                std::cout << "Actual digit: " << Y_test_lbl[sample_idx] << std::endl;
                std::cout << "Predicted digit: " << prediction << std::endl;
                std::cout << "Confidence scores:" << std::endl;
                
                for (int i = 0; i < 10; ++i) {
                    double confidence = A2s[i][0] * 100;
                    std::cout << "  Digit " << i << ": " << std::fixed << std::setprecision(2) 
                              << confidence << "%";
                    if (i == prediction) std::cout << " ← PREDICTED";
                    if (i == Y_test_lbl[sample_idx]) std::cout << " ← ACTUAL";
                    std::cout << std::endl;
                }
                
                bool correct = (prediction == Y_test_lbl[sample_idx]);
                std::cout << "Result: " << (correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "Invalid input! Please enter a number, 'draw', or -1 to exit." << std::endl;
            }
        }
    }
    
    std::cout << "\nGoodbye!" << std::endl;
    return 0;
}
