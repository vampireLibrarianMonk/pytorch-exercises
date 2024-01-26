# German Traffic Sign Recognition Benchmark GTSRB
# https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

# random: This module implements pseudo-random number generators for various distributions. It is used here to randomly
# select a subset of data (e.g., images from a dataset) for tasks like creating a validation set, shuffling data,
# or in this context, for randomly choosing images to visualize model predictions.
import random

# This line imports the 'check_output' function from the 'subprocess' module in Python. The 'subprocess' module
# allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes. This
# module is intended to replace older modules and functions like os.system and os.spawn*.
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

# PyTorch
# This line imports specific functionalities from the main PyTorch package. PyTorch is an open-source machine learning
# library extensively used for deep learning applications. It offers a flexible and powerful platform for building and
# training neural networks, with core support for multi-dimensional tensors and a wide range of mathematical operations.

# cuda: Used for operations related to CUDA, NVIDIA's parallel computing platform, enabling efficient GPU computations.
from torch import cuda

# device: Function to specify the computation device (GPU or CPU), allowing for hardware-agnostic code.
from torch import device as torch_device

# max: Function to compute the maximum value of a tensor, useful in various tensor operations and model outputs
# processing.
from torch import max as torch_max

# no_grad: This context manager is critical for inference or validation phases, where you do not want operations to
# track gradients. It reduces memory usage and speeds up computations.
from torch import no_grad

# torch.stack: This function is used to concatenate a sequence of tensors along a new dimension. All tensors need to be
# of the same size. It's often used to create a batch of tensors from multiple single instances, which is useful when
# you want to pass multiple inputs through a model simultaneously.
from torch import stack as torch_stack

# torch.save: Function for saving a serialized representation of an object (model state dictionary, entire model,
# tensors, etc.) to a file, using Python's pickle utility. It's commonly used to save trained model weights for later
# use or inference.
# Here, it's aliased as 'torch_save' for clarity and to avoid naming conflicts with other 'save' functions.
from torch import save as torch_save

# version: Provides access to the version of the PyTorch being used, useful for compatibility and debugging purposes.
from torch import version as torch_func_version

# __version__: Alias for the PyTorch version, used for checks and reporting in the development process.
from torch import __version__ as torch_version

# nn from torch: Provides the building blocks for creating neural networks, including layers, activation functions,
# and loss functions. It's a foundational module for defining and assembling neural network architectures.
import torch.nn as nn

# nn.functional from torch: Contains functions like activation functions, loss functions, and convolution operations.
# These functions are stateless, providing a functional interface to operations that can be applied to tensors.
import torch.nn.functional as F

# optim from torch: Implements various optimization algorithms for training neural networks, including SGD, Adam,
# and RMSprop. These algorithms are used to update the weights of the network during training.
import torch.optim as optim

# This line imports a package in the PyTorch library that consists of popular datasets, model architectures, and common
# image transformations for computer vision. This line imports two submodules: datasets for accessing various standard
# datasets and transforms for performing data preprocessing and augmentation operations on images.
from torchvision import datasets, transforms, __version__ as torchvision_version

# This imports the DataLoader class from PyTorch's utility functions. DataLoader is essential for loading the data and
# feeding it into the network in batches. It offers the ability to shuffle the data, load it in parallel using
# multiprocessing, and more, thus providing an efficient way to iterate over data.
from torch.utils.data import DataLoader

# matplotlib.pyplot: A collection of functions that make matplotlib work like MATLAB, used for creating static,
# interactive, and animated visualizations in Python. 'plt' is a commonly used shorthand for 'matplotlib.pyplot'.
import matplotlib.pyplot as plt

# The 'os' module in Python provides a way of using operating system dependent functionality. It allows you to interact
# with the underlying operating system in several ways, like traversing the file system, obtaining the current
# directory, performing operations on file paths, and more. It's used here to manage file paths and directories, such
# as creating a directory for data storage or checking if a directory exists.
import os

# The 'requests' library is the de facto standard for making HTTP requests in Python. It abstracts the complexities of
# making requests behind a beautiful, simple API, so that you can focus on interacting with services and consuming data
# in your application. It's used here for downloading the dataset from a given URL.
import requests

# The 'zipfile' module in Python is used for reading and writing ZIP files. It allows you to create, read, write,
# append, and list ZIP files. In this context, it's used to extract the downloaded dataset, which is assumed to be in a
# ZIP format, into a designated directory for further processing and use in the training process.
import zipfile


# SECTION 0: Version Prints
# ----------------------------------------------------------------------------------------------------------------------
# Printing software and hardware versions is crucial for:
# - Reproducibility: Ensures experiments can be replicated by providing exact environment details.
# - Debugging & Support: Facilitates troubleshooting and community assistance by identifying specific version-related
# issues.
# - Compatibility: Checks for compatibility between software libraries and hardware, especially important in ML projects
# with CUDA and GPU dependencies.
# - Performance Benchmarking: Provides context for performance results, as software/hardware updates can impact
# performance.
# - Future-Proofing & Legacy Support: Helps in maintaining the project's environment over time and ensures compatibility
# with legacy systems.
# This section prints versions of PyTorch, CUDA, and system hardware to aid in debugging, reproducibility, and support.

def version_print():
    print("Software Versions:")

    # CUDA
    if cuda.is_available():
        # Print CUDA version
        print("\tCUDA:", torch_func_version.cuda)
        device = torch_device("cuda")
    else:
        print("\tCUDA is not available.")
        device = torch_device("cpu")

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

    # TorchVision
    print("\tTorchvision:", torchvision_version)

    print("\nHardware Found:")

    # Check if CUDA is available
    if cuda.is_available():
        # Get the number of CUDA devices
        num_devices = cuda.device_count()

        print(f"\tNumber of CUDA devices available: {num_devices}")

        # Loop through all available devices
        for device_id in range(num_devices):
            # Get the name of the device
            device_name = cuda.get_device_name(device_id)
            # Get the properties of the device
            device_properties = cuda.get_device_properties(device_id)
            # Extract the total memory of the device and convert bytes to GB
            total_memory_gb = device_properties.total_memory / (1024 ** 3)

            # Print device ID, device name, and total memory in GB
            print(f"Device ID: {device_id}, Device Name: {device_name}, Total Memory: {total_memory_gb:.2f} GB")
    else:
        print("\tCUDA is not available. No devices to loop through.")


# SECTION 1: MODEL DEFINITION
# ----------------------------------------------------------------------------------------------------------------------
# This section defines the TrafficSignNet neural network architecture for traffic sign classification. The network
# consists of:
# - Two convolutional layers for feature extraction, with ReLU activations and max pooling for non-linearity and spatial
# reduction.
# - Two fully connected (dense) layers for classification, translating the high-dimensional feature representations into
# class predictions.
# - The forward pass method outlines the data flow through the network, applying each layer sequentially to produce the
# final output.
# This architecture is designed to work with RGB images and assumes a specific input size, with the output layer sized
# to match the number of traffic sign classes.

class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        # Define the first convolutional layer: 3 input channels for RGB images, 32 output channels, 3x3 kernels,
        # padding to maintain size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Define the second convolutional layer: Increase depth to 64 feature maps, 3x3 kernels, padding to maintain
        # size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Define the first fully connected layer: Flatten feature maps to vector, connect to 128 neurons
        # Adjust input size based on the size of feature maps after conv and pooling layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assumes [64, 7, 7] feature maps size before flattening

        # Define the second fully connected layer: Map 128-dimensional vector to the number of classes (e.g., 43 for
        # traffic signs)
        self.fc2 = nn.Linear(128, 43)

    def forward(self, x):
        # Apply first conv layer, ReLU activation, and 2x2 max pooling, reducing spatial dimensions
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Apply second conv layer, ReLU activation, and 2x2 max pooling, further reducing spatial dimensions
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten the feature maps into a vector for the dense layers
        x = x.view(-1, 64 * 7 * 7)

        # Apply the first dense layer with ReLU activation to introduce non-linearity
        x = F.relu(self.fc1(x))

        # Apply the second dense layer to produce class scores/logits
        x = self.fc2(x)

        # Apply log softmax to convert logits to log probabilities for each class
        return F.log_softmax(x, dim=1)


# SECTION 2: TRAINING FUNCTION
# ----------------------------------------------------------------------------------------------------------------------
# This section defines the `train_model` function, responsible for training the neural network model for one epoch. The
# process includes:
# - Setting the model to training mode to enable certain layers like Dropout and BatchNorm to behave accordingly.
# - Iterating over batches of training data, moving them to the appropriate device (GPU/CPU).
# - Performing a forward pass to compute predictions, followed by calculating the loss using the true labels.
# - Conducting a backward pass to compute gradients of the loss with respect to the model parameters.
# - Updating model parameters with an optimization step to minimize the loss.
# - Periodically printing training statistics to monitor the progress. This function encapsulates the core training
# loop, crucial for model optimization.

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()  # Switch model to training mode to activate layers like Dropout and BatchNorm correctly

    # Loop through each batch from the training data loader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Transfer data and targets to the specified device (GPU/CPU)

        optimizer.zero_grad()  # Clear previous gradients, if any, to prevent accumulation

        # Forward pass: Calculate model's predictions for the current batch of data
        output = model(data)

        # Calculate the loss by comparing the model's predictions with the actual labels
        loss = F.nll_loss(output, target)

        # Backward pass: Compute gradients of all model parameters with respect to the loss
        loss.backward()

        # Update model parameters based on gradients computed during the backward pass
        optimizer.step()

        # Log training progress every 10 batches, showing the current epoch, progress percentage, and loss
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# SECTION 3: DATASET CREATION
# ----------------------------------------------------------------------------------------------------------------------
# This section outlines the dataset preparation and transformation process, crucial for training and evaluating the
# neural network model:
# - The `download_and_extract_data` function is responsible for downloading the dataset from a specified URL and
# extracting it into a local directory, setting up the data for further processing.
# - A `preprocess` function is defined to apply a series of transformations to the input images, making them suitable
# for model processing. This includes resizing images to a uniform size, converting them to PyTorch tensors, and
# normalizing their pixel values.
# - The `create_datasets` function loads the training and validation (or test) datasets from specified directories,
# applying the preprocessing transformations.
# - The `create_data_loaders` function wraps the datasets in PyTorch DataLoader instances, facilitating efficient batch
# processing, data shuffling for the training set, and parallel data loading.

# Function to download and extract the dataset
def download_and_extract_data(url, data_path='data'):
    if not os.path.exists(data_path):
        os.makedirs(data_path)  # Create the data directory if it doesn't exist

    zip_path = os.path.join(data_path, 'dataset.zip')  # Define the full path for the zip file

    # Download the zip file from the provided URL
    response = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(response.content)  # Write the content to the zip file

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)  # Extract all the files into the data directory

    os.remove(zip_path)  # Remove the zip file to free up space


# Preprocess function adapted for PyTorch
def preprocess(img_shape=30):
    return transforms.Compose([
        transforms.Resize((img_shape, img_shape)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
    ])


def create_datasets(data_path, transform):
    # Load images from the training directory and apply preprocessing transformations
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)

    # Load images from the validation directory and apply the same transformations
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'validation'), transform=transform)

    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size=32):
    # DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, test_loader


# SECTION 4: DATASET CREATION
# ----------------------------------------------------------------------------------------------------------------------
# This section is dedicated to the testing and evaluation of the trained neural network model:
# - The `evaluate_model` function computes the model's performance metrics, such as accuracy, on the test dataset.
# This function iterates over the test dataset, computes the model's predictions, and compares them with the true labels
# to calculate accuracy or other relevant metrics.
# - Optionally, this section can include functions for visualizing predictions, confusion matrices, or other analysis
# tools that provide insights into the model's performance and behavior.
def evaluate_model(model, device, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0

    with no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability as the prediction
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)')


# Optional: Function to visualize test results or model predictions
def visualize_predictions(model, device, test_loader, num_images=5):
    model.eval()  # Set the model to evaluation mode

    # Randomly sample 'num_images' indices from the test dataset
    indices = random.sample(range(len(test_loader.dataset)), num_images)

    # Initialize lists to store sampled images and labels
    sampled_images = []
    sampled_labels = []

    # Obtain the images and labels for the randomly sampled indices
    for idx in indices:
        image, label = test_loader.dataset[idx]
        sampled_images.append(image)
        sampled_labels.append(label)

    # Stack the list of images into a batch and transfer to the device
    image_batch = torch_stack(sampled_images).to(device)

    # Generate predictions from the model by passing the batch of images through it
    output = model(image_batch)
    _, preds = torch_max(output, 1)

    # Create a new figure with specified width and height
    plt.figure(figsize=(15, 3))

    # Display each sampled image with its prediction and actual label
    for idx in range(num_images):
        ax = plt.subplot(1, num_images, idx + 1)
        img = sampled_images[idx].numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.set_title(f'Predicted: {preds[idx].item()}\nActual: {sampled_labels[idx]}')
        plt.axis('off')

    plt.show()


# SECTION 5: MAIN FUNCTION
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Print software and hardware version information
    version_print()

    # URL for the dataset to be downloaded
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/germantrafficsigns.zip'
    data_path = 'data'  # Local directory where the dataset will be stored

    # Download and extract the dataset
    download_and_extract_data(url, data_path)

    # Define the image transformations
    transform = preprocess()

    # Create the datasets
    train_dataset, test_dataset = create_datasets(data_path, transform)

    # Create the data loaders
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)

    # Set the computation device
    device = torch_device("cuda" if cuda.is_available() else "cpu")

    # Initialize the model and transfer it to the computation device
    model = TrafficSignNet().to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters())

    # Number of training epochs
    num_epochs = 10

    # Train the model
    for epoch in range(1, num_epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)

    # Evaluate the model on the test dataset
    evaluate_model(model, device, test_loader)

    # Visualize some test results
    visualize_predictions(model, device, test_loader)

    # Save the trained model
    # torch_save(model.state_dict(), "..models/traffic_sign_model.pt")


if __name__ == '__main__':
    main()
