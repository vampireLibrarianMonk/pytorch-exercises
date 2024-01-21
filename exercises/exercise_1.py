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
# ----------------------------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------------------
# Define the training function with early stopping and model checkpointing.
# This function takes the model, loss function, optimizer, data loader, patience for early stopping,
# and number of epochs as arguments.
def train_model(model, criterion, optimizer, train_loader, patience=5, n_epochs=35):
    # Initializes a variable best_loss to positive infinity. This variable will be used to keep track of the best
    # (lowest) loss during training.
    best_loss = np.inf
    # Initializes a counter variable patience_counter to 0. This variable will be used to monitor how many consecutive
    # epochs the loss does not improve.
    patience_counter = 0

    for epoch in range(n_epochs):
        # Sets the model to training mode. This is important for certain layers (e.g., dropout or batch normalization)
        # that behave differently during training and evaluation.
        model.train()
        # Initializes a variable running_loss to 0.0. This variable will accumulate the loss for each batch of training
        # data during an epoch.
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Clears the gradients from the previous step. Gradients are accumulated during the backward pass and need
            # to be reset before computing gradients for the current batch.
            optimizer.zero_grad()
            # Performs a forward pass through the neural network model (model) with the current batch of input data
            # (inputs) to obtain predictions.
            outputs = model(inputs)
            # Computes the loss between the model's predictions (outputs) and the actual target values (targets) using
            # the specified loss function (criterion).
            loss = criterion(outputs, targets)
            # Performs a backward pass to compute gradients of the loss with respect to the model's parameters. These
            # gradients will be used to update the model's weights.
            loss.backward()
            # Updates the model's weights using the optimization algorithm (optimizer) based on the computed gradients.
            optimizer.step()
            # Accumulates the loss for the current batch to running_loss.
            running_loss += loss.item()
        # Calculates the average loss for the current epoch by dividing the accumulated loss (running_loss) by the
        # number of batches in the training data.
        epoch_loss = running_loss / len(train_loader)
        # Prints the epoch number and the average loss for the current epoch.
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

        # Implement early stopping and model checkpointing
        # Checks if the epoch_loss is less than the best_loss. If it is, it updates best_loss to the current epoch_loss,
        # resets patience_counter to 0, and saves the model's state to a file named 'best_model.pth'. This is done to
        # keep track of the best model encountered during training. If epoch_loss is not better than best_loss,
        # patience_counter is incremented. If patience_counter exceeds the specified patience value, early stopping is
        # triggered by breaking out of the training loop.
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
# ----------------------------------------------------------------------------------------------------------------------
# Create a dataset using numpy arrays and convert them to PyTorch tensors.
# This dataset is a simple linear relation for demonstration purposes.
def create_dataset():
    # Initializes a variable depth with the value 10000. This variable represents the depth of the dataset and is used
    # to generate a range of training data.
    depth = 10000
    # Creates a NumPy array training_range that contains a range of values from -depth to depth. This range represents
    # the input values for the dataset.
    training_range = np.arange(-depth, depth)
    # Initializes a variable offset with the value 7. This offset is added to the training_range to generate the
    # corresponding target values.
    offset = 7
    # Creates another NumPy array test_range by adding the offset to each element in the training_range. This generates
    # the target values for the dataset.
    test_range = training_range + offset
    # Reshapes the training_range array into a 2D array with a single column using .reshape(-1, 1). Then, it converts
    # this reshaped array into a PyTorch tensor x_train with a data type of float32. x_train represents the input data
    # for the dataset.
    x_train = tensor(training_range.reshape(-1, 1), dtype=float32)
    # Reshapes the test_range array into a 2D array with a single column and converts it into a PyTorch tensor y_train
    # with a data type of float32. y_train represents the target data for the dataset.
    y_train = tensor(test_range.reshape(-1, 1), dtype=float32)
    # Combines the input data (x_train) and target data (y_train) into a PyTorch TensorDataset. This is a
    # PyTorch-specific dataset format that pairs input and target tensors for training.
    train_dataset = TensorDataset(x_train, y_train)
    # Creates a PyTorch DataLoader named train_loader using the train_dataset. This DataLoader is used to load the
    # dataset in batches during training. It has the following properties:
    #   * batch_size: Sets the batch size to 32, meaning that during training, the dataset will be divided into batches
    #   of 32 samples each.
    #   * shuffle=True: Shuffles the dataset before each epoch, ensuring that the order of data in each batch is random.
    #   This helps improve training by reducing the risk of the model memorizing the order of the data.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader


# SECTION 4: MAIN FUNCTION
# ------------------------
# The main function orchestrates the model training and evaluation.
def main():
    # Calls the create_dataset function to generate a training dataset and assigns it to the variable train_loader.
    # This DataLoader contains the training data in batches.
    train_loader = create_dataset()
    # Initializes a neural network model named model using the SimpleLinearModel class. This model is a simple linear
    # regression model with one input and one output.
    model = SimpleLinearModel()
    # Initializes the loss function criterion to be Mean Squared Error (MSE) loss. MSE is commonly used for regression
    # tasks, and it measures the average squared difference between predicted and actual values.
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    # Initializes an Adam optimizer named optimizer for updating the model's parameters during training. It is
    # configured to optimize the parameters of the model, and the learning rate (lr) is set to 0.01.
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
    # Calls the train_model function to train the model using the specified criterion, optimizer, and train_loader.
    # This function trains the model and implements early stopping.
    train_model(model, criterion, optimizer, train_loader)
    # Loads the best-trained model's state dictionary from a file named 'best_model.pth' and assigns it to the model.
    # This step is done to use the best model for predictions.
    model.load_state_dict(torch_load('best_model.pth'))

    # Sets a value for the range of input values for prediction.
    predicted_depth = 100
    # Generates a range of input values (base_x) from -predicted_depth to predicted_depth with a step of 10.
    base_x = np.arange(-predicted_depth, predicted_depth + 1, 10)
    # Converts the base_x values into a PyTorch tensor (new_x_values) with a data type of float32. These values will be used for making predictions.
    new_x_values = tensor(base_x.reshape(-1, 1), dtype=float32)
    # Sets the model to evaluation mode. This is important because some layers (e.g., dropout) behave differently during evaluation.
    model.eval()  # Set the model to evaluation mode
    # Temporarily disables gradient computation for inference to save memory and computation time.
    with no_grad():
        # Uses the trained model to make predictions for the input values in new_x_values. The predicted values are stored in predicted_y.
        predicted_y = model(new_x_values)

    # Prints the input values (base_x) and the corresponding predicted values (predicted_y) in a human-readable format.
    # The flatten() function is used to convert the predicted values from a multi-dimensional array to a flat array.
    print("Predicted y for x =", base_x, ":", predicted_y.numpy().flatten())

    # Saves the final trained model's state dictionary to a file named '../models/exercise_1_model.pth'. This file can
    # be used later for further use or deployment.
    torch_save(model.state_dict(), "../models/exercise_1_model.pth")


if __name__ == '__main__':
    main()
