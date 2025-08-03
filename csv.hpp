#ifndef CSV_HPP
#define CSV_HPP

#include <vector>
#include <string>
#include "utils.hpp"

// Reads CSV with first column = label, next 784 = pixels.
// Normalizes pixels to [0,1], returns X (784×m) and labels (size m).
void load_csv(const std::string& path,
              Matrix& X,
              std::vector<int>& Y_labels);

#endif // CSV_HPP
