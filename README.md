<h1 align="center">PyTorch Machine Learning Models</h1>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black" alt="Matplotlib">
</p>

> A foundational repository featuring end-to-end PyTorch workflows. From basic linear regression to complex Convolutional Neural Networks (CNNs), this project provides a hands-on approach to understanding modern deep learning architectures and best practices.

## 🚀 Key Features

*   **📈 Foundational Algorithms:** Implementations of baseline algorithms including Linear Regression and Binary/Multi-class Classifiers.
*   **🖼️ Computer Vision Models:** End-to-end training notebooks for Convolutional Neural Networks, notably a custom `TinyVGG` architecture.
*   **👕 FashionMNIST Integration:** Practical, real-world data handling and model evaluation using the standard FashionMNIST dataset.
*   **🛠️ Robust Helper Utilities:** A comprehensive `helper_functions.py` suite for calculating accuracy, plotting loss curves, visualizing decision boundaries, and rendering model predictions.
*   **📊 Visual Analytics:** Out-of-the-box Matplotlib integrations for rendering training/testing metrics and evaluating model convergence.

## 🏗️ Tech Stack & Architecture

This project is built primarily on the Python data science ecosystem with a heavy emphasis on **PyTorch** for tensor computations and neural network construction.

| Technology | Purpose |
| :--- | :--- |
| **Python 3.x** | Core programming language |
| **PyTorch (`torch`, `torch.nn`)** | Deep learning framework, model creation, and auto-differentiation |
| **TorchVision** | Computer vision datasets, model architectures, and image transformations |
| **Jupyter Notebook** | Interactive development environment for model training and experimentation |
| **Matplotlib / NumPy** | Data visualization, metric plotting, and array manipulations |

**Architecture Flow:**
1.  **Data Loading:** Datasets (like FashionMNIST) are downloaded, transformed via `torchvision.transforms`, and batched using `DataLoader`.
2.  **Model Definition:** Models subclass `torch.nn.Module`, defining the `__init__` layers and the `forward` pass.
3.  **Training Loop:** Iterative optimization using loss functions (e.g., `CrossEntropyLoss`, `L1Loss`) and optimizers (e.g., `SGD`, `Adam`).
4.  **Evaluation & Visualization:** The `helper_functions.py` script abstracts away the complex matplotlib code to visualize decision boundaries, loss curves, and actual vs. predicted images.

## 🏁 Getting Started

### Prerequisites

Ensure you have the following installed on your system:
*   Python 3.8 or higher
*   Git
*   (Optional but recommended) A CUDA-compatible GPU for faster training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/pytorch-ml-models.git
    cd pytorch-ml-models
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio
    pip install matplotlib numpy jupyter requests
    ```

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

## 💻 Usage

Once Jupyter is running, open any of the `.ipynb` files to run the models. The notebooks are structured sequentially to guide you through data preparation, model building, training, and evaluation.

**Example: Visualizing Model Predictions**

You can leverage the built-in helper functions within your notebooks to easily visualize how well your trained model is performing:

```python
import torch
from helper_functions import plot_loss_curves, pred_and_plot_image

# Assuming 'results' is a dictionary generated during your training loop
plot_loss_curves(results)

# Make a prediction on a single image and visualize it
pred_and_plot_image(
    model=trained_model,
    image_path="data/test_image.jpg",
    class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

## 📂 Project Structure

```text
pytorch-ml-models/
│
├── data/                                         # Directory for downloaded datasets (e.g., FashionMNIST)
├── models/                                       # Directory for saving trained model state_dicts (.pth)
│
├── helper_functions.py                           # Core utility scripts for plotting, metrics, and data downloading
│
├── Linear_Regression_Model.ipynb                 # Basics of PyTorch tensors and linear regression
├── Binary_Classifier_Model.ipynb                 # Neural network for binary classification tasks
├── MultiClass_Classifier_Model.ipynb             # Neural network for multi-class classification tasks
├── FMNIST_Linear_Model.ipynb                     # Applying a linear model to the FashionMNIST dataset
├── FMNIST_Convolutional_Neural_Network_Model.ipynb # Applying a CNN to the FashionMNIST dataset
└── TinyVGG_model.ipynb                           # Advanced CNN architecture implementation (TinyVGG)
```




