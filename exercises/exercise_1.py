# This line imports specific functionalities from the main PyTorch package. PyTorch is an open-source machine learning
# library extensively used for deep learning applications. It offers a flexible and powerful platform for building and
# training neural networks, with core support for multi-dimensional tensors and a wide range of mathematical operations.

# The specific components imported are:
# 1. **cuda**: This module is used for all things related to CUDA, NVIDIA's parallel computing platform and API model.
#    It allows for direct interaction with the GPU for efficient computation, especially beneficial for deep learning
#    tasks.
# 2. **device**: This function is utilized to set up the device on which to perform computations
# (e.g., 'cuda' or 'cpu').
#    Using 'device', one can write hardware-agnostic code that automatically utilizes GPUs if they're available.
# 3. **flatten**: A function to flatten a tensor. It collapses a multidimensional tensor into a single dimension,
#    often used in transitioning from convolutional layers to fully connected layers in a neural network.
# 4. **no_grad**: This context manager is critical for inference or validation phases, where you do not want operations
#    to track gradients. It reduces memory usage and speeds up computations.
# 5. **__version__ as torch_version**: This imports the version identifier of the PyTorch library (aliased as
#    'torch_version'). It's helpful for compatibility checks or when reporting issues.
from torch import (cuda, device, float32, load as torch_load, no_grad, save as torch_save, tensor,
                   version as torch_func_version, __version__ as torch_version)

# This line imports the nn module from PyTorch, aliased as nn. The nn module provides a way of defining a neural
# network. It includes all the building blocks required to create a neural network, such as layers, activation
# functions, and other utilities.
import torch.nn as nn

# This line imports the optim module, aliased as optim. The optim module includes various optimization algorithms that
# can be used to update the weights of the network during training. Common optimizers like Stochastic Gradient Descent
# (SGD), Adam, and RMSprop are included in this module.
import torch.optim as optim

# This imports the DataLoader and TensorDataset classes from PyTorch's utility functions. DataLoader is essential for
# loading the data and feeding it into the network in batches. It offers the ability to shuffle the data, load it in
# parallel using multiprocessing, and more, thus providing an efficient way to iterate over data. TensorDataset is a
# dataset wrapping tensors. By defining a dataset of tensors, you can easily index and access the data for training
# and evaluation. When combined, TensorDataset and DataLoader provide a flexible way to feed data into your model.
from torch.utils.data import DataLoader, TensorDataset

# This line imports the NumPy library and aliases it as 'np'. NumPy, which stands for Numerical Python, is a fundamental
# package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices,
# along with a collection of high-level mathematical functions to operate on these arrays. The alias 'np' is a widely
# adopted convention used for the sake of brevity and convenience in the code.

# Key Features of NumPy:
# 1. **Efficient Array Processing**: At the core of NumPy is the 'ndarray' object, an efficient multi-dimensional array
#    providing fast array-oriented arithmetic operations and flexible broadcasting capabilities.
# 2. **Mathematical Functions**: NumPy offers a vast range of mathematical functions such as linear algebra operations,
#    Fourier transforms, and random number generation, which are essential for various scientific computing tasks.
# 3. **Interoperability**: NumPy arrays can easily be used as inputs for other libraries like SciPy, Matplotlib, and
#    Pandas, making it a foundational library in the Python data science and machine learning ecosystem.
# 4. **Performance**: Written primarily in C, NumPy operations are executed much more efficiently than standard Python
#    sequences, especially for large data sets. This makes it a preferred choice for data-intensive computations.

# Common Usage:
# NumPy is extensively used in domains like data analysis, machine learning, scientific computing, and engineering for
# tasks such as data transformation, statistical analysis, and image processing. The 'np' alias simplifies the access to
# NumPy functions, allowing for concise and readable code.
import numpy as np

# Key aspects of 'check_output':
# 1. **Process Execution**: The 'check_output' function is used to run a command in the subprocess/external process and
#    capture its output. This is especially useful for running system commands and capturing their output directly
#    within a Python script.
# 2. **Return Output**: It returns the output of the command, making it available to the Python environment. If the
#    called command results in an error (non-zero exit status), it raises a CalledProcessError.
# 3. **Use Cases**: Common use cases include executing a shell command, reading the output of a command, automating
#    scripts that interact with the command line, and integrating external tools into a Python workflow.

# Example Usage:
# Suppose you want to capture the output of the 'ls' command in a Unix/Linux system. You can use 'check_output' like
# this:
# output = check_output(['ls', '-l'])
from subprocess import check_output

# SECTION 0: Version Prints
# This line imports the 'check_output' function from the 'subprocess' module in Python. The 'subprocess' module allows
# you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes. This module is
# intended to replace older modules and functions like os.system and os.spawn*.

print("Software Versions:")

# CUDA
if cuda.is_available():
    # Print CUDA version
    print("\tCUDA:", torch_func_version.cuda)
    device = device("cuda")
else:
    print("\tCUDA is not available.")
    device = device("cpu")

# NVIDIA Driver
try:
    # Execute the nvidia-smi command and decode the output
    nvidia_smi_output = check_output("nvidia-smi", shell=True).decode()

    # Split the output into lines
    lines = nvidia_smi_output.split('\n')

    # Find the line containing the driver version
    driver_line = next((line for line in lines if "Driver Version" in line), None)

    # Extract the driver version number
    if driver_line:
        driver_version = driver_line.split('Driver Version: ')[1].split()[0]
        print("\tNVIDIA Driver:", driver_version)

        # Extract the maximum supported CUDA version
        cuda_version = driver_line.split('CUDA Version: ')[1].strip().replace("|", "")
        print("\tMaximum Supported CUDA Version:", cuda_version)
    else:
        print("\tNVIDIA Driver Version or CUDA Version not found.")

except Exception as e:
    print("Error fetching NVIDIA Driver Version or CUDA Version:", e)

# Torch
print("\tPyTorch:", torch_version)

print("\n")


# SECTION 1: MODEL DEFINITION
# ---------------------------
# Define a simple linear model by subclassing nn.Module.
# This model has a single linear layer with one input and one output.
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        # The self.linear(x) call applies a linear transformation to the input data x. This transformation is defined
        # by the linear layer's weights and biases, which are automatically learned during the training process.
        # The linear layer essentially performs the operation y = wx + b, where w is the weight, x is the input,
        # b is the bias, and y is the output.
        self.linear = nn.Linear(1, 1)

    # Defines the computation performed at every call of the neural network.
    def forward(self, x):
        return self.linear(x)


# SECTION 2: TRAINING FUNCTION
# ----------------------------
# Define the training function with early stopping and model checkpointing.
# This function takes the model, loss function, optimizer, data loader, patience for early stopping,
# and number of epochs as arguments.
def train_model(model, criterion, optimizer, train_loader, patience=5, n_epochs=35):
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients from the previous step
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item()  # Accumulate the loss

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        # Implement early stopping and model checkpointing
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch_save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


# SECTION 3: DATASET CREATION
# ---------------------------
# Create a dataset using numpy arrays and convert them to PyTorch tensors.
# This dataset is a simple linear relation for demonstration purposes.
def create_dataset():
    depth = 10000
    training_range = np.arange(-depth, depth)
    offset = 7
    test_range = training_range + offset

    x_train = tensor(training_range.reshape(-1, 1), dtype=float32)
    y_train = tensor(test_range.reshape(-1, 1), dtype=float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader


# SECTION 4: MAIN FUNCTION
# ------------------------
# The main function orchestrates the model training and evaluation.
def main():
    train_loader = create_dataset()
    model = SimpleLinearModel()
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

    train_model(model, criterion, optimizer, train_loader)

    # Load the best model for prediction
    model.load_state_dict(torch_load('best_model.pth'))

    # Prediction
    predicted_depth = 100
    base_x = np.arange(-predicted_depth, predicted_depth + 1, 10)
    new_x_values = tensor(base_x.reshape(-1, 1), dtype=float32)
    model.eval()  # Set the model to evaluation mode
    with no_grad():  # Disable gradient computation for inference
        predicted_y = model(new_x_values)

    print("Predicted y for x =", base_x, ":", predicted_y.numpy().flatten())

    # Save the final model
    torch_save(model.state_dict(), "../models/exercise_1_model.pth")


if __name__ == '__main__':
    main()
