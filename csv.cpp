#include "csv.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

void load_csv(const std::string& path,
              Matrix& X,
              std::vector<int>& Y_labels) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    std::string line;

    if (!std::getline(in, line)) {
        throw std::runtime_error("CSV file is empty: " + path);
    }

    std::vector<std::vector<double>> data;
    int line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        try {
            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stod(cell));
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: skipping malformed line " << line_no
                      << " in CSV: \"" << line << "\"\n";
            continue;
        }
        if (row.size() != 785) {
            std::cerr << "Warning: skipping line " << line_no
                      << " (expected 785 columns, got " << row.size() << ")\n";
            continue;
        }
        data.push_back(std::move(row));
    }

    int m = data.size();
    if (m == 0) {
        throw std::runtime_error("No valid data rows found in CSV.");
    }

    X.resize(784, std::vector<double>(m));
    Y_labels.resize(m);

    for (int j = 0; j < m; ++j) {
        Y_labels[j] = static_cast<int>(data[j][0]);
        for (int i = 0; i < 784; ++i) {
            X[i][j] = data[j][i + 1];
        }
    }
}
