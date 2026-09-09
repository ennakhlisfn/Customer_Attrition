# Customer_Attrition

A deep learning binary classification pipeline built with PyTorch to predict customer churn. This project processes numerical feature sets, handles severe class imbalance, and trains a neural network using mini-batch gradient descent.

## Features

* **Data Preprocessing:** Implements standard feature scaling (`StandardScaler`) to prevent gradient explosion and optimize learning.
* **Mini-Batch Training:** Utilizes PyTorch's `TensorDataset` and `DataLoader` for efficient batch processing and stochastic gradient updates.
* **Class Imbalance Handling:** Automatically calculates target class distributions and applies a `pos_weight` to the `BCEWithLogitsLoss` function to prevent the model from ignoring the minority class (churners).
* **Memory Efficiency:** Converts standard 64-bit NumPy arrays to 32-bit PyTorch Tensors (`float32`) for faster GPU/CPU processing.

## Prerequisites

Ensure you have the following libraries installed:
* `torch`
* `scikit-learn`
* `pandas`
* `numpy`

## Usage

### 1. Data Preparation
The pipeline expects a feature set `X` and a target variable `y` (binary labels `0` or `1`). The data is split into an 80/20 train-test ratio and scaled using statistics strictly learned from the training set to prevent data leakage.

### 2. Training the Model
The model (`ChurnModel`) is trained using the Adam optimizer. Hyperparameters such as batch size and the number of epochs should be tuned based on the dataset size and complexity. 

* **Recommended starting parameters:** `batch_size=32` or `64`, `epochs=10` to `120`. 
* *Note on Overfitting:* If the validation accuracy drops while training loss decreases, consider reducing the number of epochs (e.g., stopping around 10-20 epochs) or implementing early stopping.

```python
# Example initialization
model = ChurnModel(input_dim=X_train_tensor.shape[1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
optimizer = optim.Adam(model.parameters(), lr=0.01)
