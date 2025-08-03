# Neural Network CPP

A feedforward neural network implementation from scratch in C++ for MNIST handwritten digit recognition. The network features a two-layer architecture with ReLU activation and softmax output, achieving ~84% accuracy on the test set.

## Features

- **MNIST digit recognition** (0-9 classification)
- **Interactive prediction mode** with confidence scores
- **Draw your own digits** using ASCII art (28x28 grid)
- **Dual prediction modes**: Test existing samples or draw custom digits
- **Gradient clipping** for training stability
- **He weight initialization** optimized for ReLU activations
- **Mini-batch gradient descent** with learning rate decay and OpenMP parallelized matrix operations
- **Comprehensive performance metrics** (precision, recall, F1-score per class)

## Prerequisites

- C++11 compatible compiler (e.g., `g++`) with OpenMP support
- CMake 3.0 or higher
- OpenMP library (automatically detected by CMake)
- MNIST dataset (CSV format)

## Directory Structure

```
neural_network_cpp/
│
├── build/                # Build directory
├── csv.cpp               # CSV parsing utilities
├── csv.hpp
├── digit_recognizer.csv  # MNIST dataset in CSV format must be placed here
├── main.cpp              # Main program with training and interactive mode
├── neural_network.cpp    # Neural network implementation
├── neural_network.hpp
├── utils.cpp             # Utility functions for matrix operations
├── utils.hpp
└── CMakeLists.txt        # CMake configuration
```

## Build Instructions

1. **Navigate to the project directory:**
   ```sh
   cd /path/to/neural_network_cpp
   ```

2. **Create a build directory and navigate into it:**
   ```sh
   mkdir -p build && cd build
   ```

3. **Run CMake to configure the build system:**
   ```sh
   cmake ..
   ```

4. **Build the project with OpenMP support:**
   ```sh
   make
   ```

## Running the Neural Network

1. **Ensure the MNIST dataset file `digit_recognizer.csv` is available in the root directory.**

2. **Execute the neural network binary:**
   ```sh
   ./neural_network
   ```

   The program will:
   - Load the dataset (42,000 samples)
   - Split into 80% training, 20% testing
- Train for 8 epochs with mini-batch gradient descent and OpenMP-optimized matrix operations
   - Display training progress with loss and accuracy
   - Show overall test performance metrics

3. **Interactive Prediction Mode:**
   
   **Mode 1: Test Existing Samples**
   - Enter a sample index (0-8399) to see detailed prediction
   - View confidence scores for all 10 digits (0-9)
   - See actual vs predicted digit with correctness indicator
   
   **Mode 2: Draw Your Own Digit**
   - Enter `draw` to activate drawing mode
   - Draw a 28x28 ASCII art digit using the provided templates
   - Get real-time prediction on your custom drawing
   
   **Exit**
   - Enter `-1` to exit the program

## Example Output

```
Loading dataset...
Loaded 42000 samples.
Epoch 1/10 — loss: 1.08101 — acc:  0.781071
Epoch 2/10 — loss: 0.802601 — acc:  0.826399
Epoch 3/10 — loss: 0.715090 — acc:  0.838631
...
Overall Accuracy: 0.845000

==================================================
INTERACTIVE PREDICTION MODE
==================================================
Enter a sample index (0-8399) to see prediction, or -1 to exit:
25

--- Sample #25 ---
Actual digit: 4
Predicted digit: 4
Confidence scores:
  Digit 0: 0.27%
  Digit 1: 0.41%
  Digit 2: 0.78%
  Digit 3: 6.02%
  Digit 4: 73.80% ← PREDICTED ← ACTUAL
  Digit 5: 5.04%
  Digit 6: 5.06%
  Digit 7: 0.91%
  Digit 8: 5.16%
  Digit 9: 2.55%
Result: ✓ CORRECT
```

## Network Architecture

- **Input Layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation (digit probabilities)
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Mini-batch gradient descent (batch size: 64)
- **Learning Rate**: 0.003 with decay every 10 epochs
- **Training**: 8 epochs with OpenMP for faster computation
- **Gradient Clipping**: ±1.0 for training stability

## Configuration

Key parameters can be modified in `main.cpp`:

```cpp
int hidden_units = 128;        // Hidden layer size
double learning_rate = 0.003;  // Learning rate
int batch_size = 64;           // Mini-batch size
int epochs = 8;               // Training epochs
```

## OpenMP Optimization

This implementation leverages OpenMP for parallel matrix multiplication, significantly improving training performance:

- **Matrix Operations**: The core `matmul` function uses OpenMP `#pragma omp parallel for collapse(2)` to parallelize nested loops
- **Automatic Scaling**: Utilizes all available CPU cores automatically
- **Conditional Compilation**: OpenMP features are enabled only when the library is available
- **Performance Gain**: Achieves 2-3x speedup on multi-core systems during training and inference

### OpenMP Detection
The build system automatically detects and enables OpenMP if available. You can verify OpenMP is working by observing faster training times and higher CPU utilization during execution.

## Drawing Your Own Digits

### How to Draw

1. **Enter drawing mode** by typing `draw` when prompted
2. **Use the character guide:**
   - `#`, `*`, `@` for dark pixels (the digit)
   - `.` for light pixels (background)
   - `+`, `o` for medium gray pixels
3. **Draw 28 lines** of exactly 28 characters each
4. **Use templates** from `digit_templates.txt` for reference

### Example Digit 7:
```
............................
....##################......
......................##....
.....................##.....
....................##......
...................##.......
..................##........
.................##.........
................##..........
...............##...........
..............##............
.............##.............
............##..............
...........##...............
..........##................
.........##.................
........##..................
.......##...................
......##....................
.....##.....................
............................
```

### Tips for Better Recognition
- Make digits **thick and clear**
- Center the digit in the 28x28 grid
- Use consistent stroke width
- Refer to `digit_templates.txt` for examples

## Dataset Requirements

The MNIST dataset should be in CSV format with:
- First column: digit label (0-9)
- Remaining 784 columns: pixel values (0-255)
- Header row (automatically skipped)
- File name: `digit_recognizer.csv`

## Performance

- **Training Time**: ~12-15 seconds for 8 epochs (with OpenMP parallelization)
- **Test Accuracy**: ~84-87% (optimized training parameters)
- **Memory Usage**: Efficient matrix operations with OpenMP parallel computing
- **Training Stability**: Gradient clipping prevents exploding gradients
- **Speedup**: 2-3x faster matrix operations using multiple CPU cores

## Troubleshooting

**Build Issues:**
- Ensure C++11 support: `g++ -std=c++11`
- Check CMake version: `cmake --version`
- Verify OpenMP support: `echo | cpp -fopenmp -dM | grep -i openmp`

**Runtime Issues:**
- Verify `digit_recognizer.csv` exists in project root
- Check file permissions and format
- Monitor memory usage for large datasets
