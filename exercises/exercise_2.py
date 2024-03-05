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

# Importing the PyTorch library, known as `torch`, a powerful and widely used open-source machine learning framework.
# PyTorch provides tools and libraries for designing, training, and deploying deep learning models with ease. It's
# particularly known for its flexibility, user-friendly interface, and dynamic computational graph that allows for
# adaptive and efficient deep learning development. By importing `torch`, you gain access to a vast range of
# functionalities for handling multi-dimensional arrays (tensors), performing complex mathematical operations,
# and utilizing GPUs for accelerated computing. This makes it an indispensable tool for both researchers and
# developers in the field of artificial intelligence.
import torch

# This line imports the nn module from PyTorch, aliased as nn. The nn module provides a way of defining a neural
# network. It includes all the building blocks required to create a neural network, such as layers, activation
# functions, and other utilities.
import torch.nn as nn

# This line imports the optim module, aliased as optim. The optim module includes various optimization algorithms that
# can be used to update the weights of the network during training. Common optimizers like Stochastic Gradient Descent
# (SGD), Adam, and RMSprop are included in this module.
import torch.optim as optim

# This imports the DataLoader class from PyTorch's utility functions. DataLoader is essential for loading the data and
# feeding it into the network in batches. It offers the ability to shuffle the data, load it in parallel using
# multiprocessing, and more, thus providing an efficient way to iterate over data.
from torch.utils.data import DataLoader

# A toolbox for computer vision tasks in Python. It includes:
# 1. 'datasets' for accessing a variety of pre-prepared image collections for training models.
# 2. 'transforms' for editing and adjusting images (like resizing or color changes) to improve model learning.
import torchvision

# This imports the StepLR class from the lr_scheduler submodule in optim. Learning rate schedulers adjust the learning
# rate during training, which can lead to more effective and faster training. StepLR decreases the learning rate at
# specific intervals, which is helpful in fine-tuning the network as training progresses.
from torch.optim.lr_scheduler import StepLR


# SECTION 1: MODEL DEFINITION
# ----------------------------------------------------------------------------------------------------------------------
# Define the neural network architecture
# * nn.Conv2d: A 2D convolutional layer that performs a convolution operation on the input image. It extracts features
#   from the input image.
# * nn.BatchNorm2d: Is applied after the convolutional layer. It normalizes the activations of the previous layer,
#   helping in training stability and faster convergence.
# * nn.Dropout: Randomly sets a fraction of input units to zero during training (25% in this case). It helps prevent
#   overfitting by introducing noise and encouraging the network to learn more robust features.
# * nn.Linear: Is a fully connected (linear) layer that takes the flattened output from the previous layers (3x3x32)
#   and produces the final classification output with num_classes output units. It performs the final classification
#   based on learned features.
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        # This initializes the parent class (nn.Module) for our custom neural network model.
        # It sets up the basic structure and functionalities of the network.
        super(FashionMNISTModel, self).__init__()

        # Define the first convolutional layer:
        # - 1 input channel (e.g., grayscale image)
        # - 128 output channels (number of filters)
        # - Kernel size of 3 (3x3 filter)
        # - Padding of 1 to keep the spatial dimensions of the output same as the input
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)

        # Batch normalization for the first convolutional layer:
        # - Normalizes the output of the previous layer to improve training speed and stability
        # - 128 features to match the output channels of the first convolutional layer
        self.bn1 = nn.BatchNorm2d(128)

        # Define the second convolutional layer:
        # - 128 input channels (from the previous conv layer)
        # - 64 output channels
        # - Kernel size of 3 (3x3 filter)
        # - Padding of 1 for consistent spatial dimensions
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Batch normalization for the second convolutional layer:
        # - 64 features to match the output channels of the second convolutional layer
        self.bn2 = nn.BatchNorm2d(64)

        # Define the third convolutional layer:
        # - 64 input channels (from the previous conv layer)
        # - 32 output channels
        # - Kernel size of 3 (3x3 filter)
        # - Padding of 1 for consistent spatial dimensions
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # Batch normalization for the third convolutional layer:
        # - 32 features to match the output channels of the third convolutional layer
        self.bn3 = nn.BatchNorm2d(32)

        # Dropout layer with a dropout probability of 25% to prevent overfitting.
        self.dropout1 = nn.Dropout(0.25)

        # Flatten layer to transform the multidimensional output of the conv layers into
        # a one-dimensional tensor suitable for input into the fully connected layer.
        self.flatten = nn.Flatten()

        # Dropout layer with a higher dropout probability of 50%.
        self.dropout2 = nn.Dropout(0.5)

        # Define a fully connected (linear) layer to output the final class predictions:
        # - Inputs are flattened feature maps from the conv layers
        # - Output size is 'num_classes' for classification
        # - The '3 * 3 * 32' is calculated based on the output size of the last convolutional layer
        self.fc1 = nn.Linear(3 * 3 * 32, num_classes)

    # A forward pass in PyTorch involves passing input data through the neural network to compute predictions, which are
    #   then compared to the true values to calculate a loss. This loss is essential for updating the model's parameters
    #   and improving its performance during training. The forward pass is a fundamental step in the training process
    #   of neural networks, enabling them to learn from data and make predictions.

    # Here's a breakdown of what happens during a forward pass in a PyTorch training model:
    # * Input Data: The forward pass begins with providing input data to the neural network. This input data could be a
    #   single data point or a batch of data points, depending on the chosen batch size.

    # * Layer Computations: As the input data moves through the network, it sequentially passes through various layers
    #   of the neural network, including convolutional layers, activation functions, fully connected layers, and more.
    #   Each layer performs specific computations on the input data.

    # * Activation Functions: Activation functions, such as ReLU (Rectified Linear Unit), sigmoid, or tanh, introduce
    #   non-linearity to the network. They determine whether a neuron should be activated or not based on its input.
    #   ReLU, for example, replaces negative values with zero and allows positive values to pass through.

    # * Weighted Sum: In fully connected layers, each neuron computes a weighted sum of its inputs, which are the
    #   outputs from the previous layer. These weights are learned during the training process and represent the
    #   network's knowledge.

    # * Output Layer: The final layer of the network, often referred to as the output layer, produces the network's
    #   predictions. The type of task (e.g., classification, regression) determines the activation function used in the
    #   output layer.

    # * Loss Calculation: Once the predictions are obtained, they are compared to the ground truth (the actual target
    #   values) to calculate a loss or error value. Common loss functions include Mean Squared Error (MSE) for
    #   regression tasks and Cross-Entropy Loss for classification tasks.

    # * Backpropagation: After the forward pass, the network performs a backward pass (backpropagation) to compute
    #   gradients of the loss with respect to the network's parameters. These gradients are used to update the model's
    #   weights during training, optimizing the network's performance.

    # * Backpropagation is a key algorithm used in training artificial neural networks, allowing them to learn from data
    # . It involves calculating gradients of the loss function with respect to the network's parameters by propagating
    # errors backward through the network. These gradients guide the adjustment of model parameters during training,
    # leading to the gradual improvement of the network's ability to make accurate predictions.

    def forward(self, x):
        # Convolutional Layer 1
        x = self.conv1(x)  # Apply the first convolutional layer to the input
        x = nn.functional.relu(x)  # Apply the rectified linear unit (ReLU) activation function
        x = self.bn1(x)  # Apply batch normalization to normalize the activations
        x = nn.functional.max_pool2d(x, 2)  # Apply 2x2 max pooling to downsample the feature maps
        x = self.dropout1(x)  # Apply dropout with a 25% probability of setting units to zero

        # Convolutional Layer 2
        x = self.conv2(x)  # Apply the second convolutional layer to the output of the first layer
        x = nn.functional.relu(x)  # Apply ReLU activation
        x = self.bn2(x)  # Apply batch normalization
        x = nn.functional.max_pool2d(x, 2)  # Apply max pooling
        x = self.dropout1(x)  # Apply dropout (same as before, should be dropout2)

        # Convolutional Layer 3
        x = self.conv3(x)  # Apply the third convolutional layer
        x = nn.functional.relu(x)  # Apply ReLU activation
        x = self.bn3(x)  # Apply batch normalization
        x = nn.functional.max_pool2d(x, 2)  # Apply max pooling

        # Flatten Layer
        x = torch.flatten(x, 1)  # Flatten the 2D feature maps into a 1D vector

        # Dropout Layer 2
        x = self.dropout2(x)  # Apply a different dropout layer with 50% probability

        # Fully Connected Layer (Output)
        x = self.fc1(x)  # Apply the fully connected layer to produce the final output
        output = nn.functional.softmax(x, dim=1)  # Apply softmax to obtain class probabilities
        return output


# SECTION 2: TRAINING FUNCTION
# ----------------------------------------------------------------------------------------------------------------------
# Training loop
# During training, the model learns from the data to make accurate predictions. This loop processes batches of training
# data, computes the loss between predictions and actual labels, and updates the model's parameters to minimize this
# loss. It's a fundamental process in machine learning and deep learning, allowing models to improve their performance
# over time through optimization.
def train(model, device, train_loader, optimizer, epoch, criterion):
    # Sets the model to training mode. Important for layers like dropout and batch normalization to work correctly
    # during training.
    model.train()

    # Calculate the total number of batches in the train_loader to determine progress intervals.
    total_batches = len(train_loader)
    # Calculate what constitutes 10% of total batches for progress updates.
    ten_percent_batches = total_batches / 10
    # Create a list of batch indices where each represents a 10% progress milestone.
    milestones = [int(ten_percent_batches * i) for i in range(1, 11)]

    # Start iterating over the training data in batches.
    for batch_idx, (data, target) in enumerate(train_loader):
        # Transfer the data and target to the current device (GPU or CPU), ensuring compatibility.
        data, target = data.to(device), target.to(device)
        # Reset gradients to zero before starting to do backpropragation because gradients accumulate by default.
        optimizer.zero_grad()
        # Forward pass: compute the model output for the current batch of data.
        output = model(data)
        # Compute the loss between the model's predictions and the actual target values.
        loss = criterion(output, target)
        # Backward pass: compute gradient of the loss with respect to model parameters.
        loss.backward()
        # Perform a single optimization step (parameter update).
        optimizer.step()

        # Check if the current batch index matches any of the predefined milestones for 10% increments.
        if batch_idx in milestones:
            # Print the training progress, including epoch, number of samples processed, total number of samples,
            # percentage of progress, and the current loss value.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# SECTION 3: Testing Function
# ----------------------------------------------------------------------------------------------------------------------
# Testing loop
# Evaluates the model's performance on a separate dataset not used for training. During testing, the model is set to
# evaluation mode, and it makes predictions on the test data. The code calculates the average loss and accuracy of the
# model's predictions, providing valuable insights into its performance. It's a crucial step in assessing how well the
# trained model generalizes to unseen data and is essential for model evaluation and validation.
def test(model, device, test_loader, criterion):
    # Sets the model into evaluation mode. During evaluation, the model behaves differently from training mode. For
    # example, layers like dropout and batch normalization may work differently.
    model.eval()
    # Keep track of the total loss and the number of correct predictions made during the testing process.
    test_loss = 0
    correct = 0
    # This context manager is used to temporarily disable gradient calculation. During testing, we don't need to compute
    # gradients because we are not updating the model's weights. This can significantly speed up the evaluation process
    with torch.no_grad():
        # Iterates through the test dataset using the data loader. It loads a batch of test data along with their
        # corresponding target labels.
        for data, target in test_loader:
            # Moves both the input data and target labels to the device (CPU or GPU) used for computation. This ensures
            # that calculations are performed on the selected device.
            data, target = data.to(device), target.to(device)
            # Model is used to make predictions on the test data, resulting in the output, which contains predicted
            # class scores for each sample in the batch.
            output = model(data)
            # Computes the loss between the model's predictions (output) and the ground truth labels (target). This loss
            # is added to the test_loss variable to keep track of the cumulative loss over the entire test dataset.
            test_loss += criterion(output, target).item()
            # For each sample in the batch, this line determines the class with the highest predicted score. The pred
            # tensor contains the predicted class labels.
            pred = output.argmax(dim=1, keepdim=True)
            # Calculates the number of correct predictions by comparing the predicted labels (pred) with the true labels
            # (target). It counts how many predictions match the ground truth labels and adds this count to the correct
            # variable.
            correct += pred.eq(target.view_as(pred)).sum().item()
    # After processing all batches, test_loss is divided by the total number of samples in the test dataset to calculate
    # the average loss.
    test_loss /= len(test_loader.dataset)
    # Prints the test results, including the average loss and accuracy. The accuracy is the percentage of correctly
    # classified samples out of the total test dataset.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# SECTION 4: CREATE DATASET
# ----------------------------------------------------------------------------------------------------------------------
def create(device):
    # Load the FashionMNIST dataset
    # * transforms.Compose: This line initializes a sequence of data transformations to be applied to the dataset. These
    #   transformations will modify the dataset's images before they are used for training or testing.
    # * transforms.ToTensor: This transformation converts the images in the dataset from their original format
    # (typically stored as pixel values in a range from 0 to 255) into PyTorch tensors. Tensors are the primary data
    # structure used for deep learning in PyTorch, and this transformation ensures that the images are represented as \
    # tensors.
    # * transforms.Normalize: This normalization transformation is applied to the tensorized images. It
    #   subtracts the mean value o from each pixel and then divides by the standard deviation. This process scales the
    #   pixel values to be in the range of -1 to 1, which can help improve the training of neural networks by making the
    #   data more centered around zero.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # Load and prepare the FashionMNIST dataset for both training and testing in a PyTorch-based deep learning model.
    # * Arg 1: This string specifies the directory where the dataset will be stored or is already located. If the
    # dataset is not present in this directory, it will be downloaded automatically due to the download=True argument.
    # * train=True: This argument indicates that we are creating a dataset for training, so it will include the training
    #   images and labels. When set to False, it would create a dataset for testing, including test images and labels.
    # * transform=transform: Dataset images will undergo the transformations defined in transform, including converting
    # them to tensors and normalizing their pixel values.
    train_dataset = torchvision.datasets.FashionMNIST('../data',
                                          train=True,
                                          download=True,
                                          transform=transform
                                          )
    test_dataset = torchvision.datasets.FashionMNIST('../data',
                                         train=False,
                                         transform=transform
                                         )

    # Data loaders will be used during the training and testing phases of a deep learning model. They allow for
    # efficient batching and optional shuffling of data, making it easier to train and evaluate the model.
    # * This argument specifies the dataset that the data loader will work with, which, in this case, is the
    # FashionMNIST training dataset.
    # * batch_size=32: The batch_size argument determines the number of data samples (images) that will be processed
    #   together in each iteration. In this case, a batch size of 32 is chosen, meaning that 32 images will be processed
    #   in parallel during each training iteration.
    # * Setting shuffle to True indicates that the data loader will shuffle the order of the data samples in each epoch.
    #   Shuffling is a common practice in training deep learning models to ensure that the model doesn't learn the order
    #   of the data. This helps in achieving better generalization.
    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True
                              )
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False
                             )

    # Initialize the model, optimizer, and loss function
    model = FashionMNISTModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize the StepLR scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    return transform, train_dataset, test_dataset, train_loader, test_loader, model, optimizer, criterion, scheduler


# SECTION 5: MAIN FUNCTION
# ----------------------------------------------------------------------------------------------------------------------
# The main function orchestrates the model training and evaluation.
def main():
    # SECTION 0: Version Prints
    print("Software Versions:")

    # CUDA
    if torch.cuda.is_available():
        # Print CUDA version
        print("\tCUDA:", torch.version.cuda)
        device = torch.device("cuda")
    else:
        print("\tCUDA is not available.")
        device = torch.device("cpu")

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
    print("\tPyTorch:", torch.version)

    # TorchVision
    print("\tTorchvision:", torchvision.version)

    print("Hardware Found:")

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of CUDA devices
        num_devices = torch.cuda.device_count()

        print(f"\tNumber of CUDA devices available: {num_devices}")

        # Loop through all available devices
        for device_id in range(num_devices):
            # Get the name of the device
            device_name = torch.cuda.get_device_name(device_id)
            # Get the properties of the device
            device_properties = torch.cuda.get_device_properties(device_id)
            # Extract the total memory of the device and convert bytes to GB
            total_memory_gb = device_properties.total_memory / (1024 ** 3)

            # Print device ID, device name, and total memory in GB
            print(f"Device ID: {device_id}, Device Name: {device_name}, Total Memory: {total_memory_gb:.2f} GB")
    else:
        print("\tCUDA is not available. No devices to loop through.")

    print("\n")

    # Create and get dataset
    (transform,
     train_dataset,
     test_dataset,
     train_loader,
     test_loader,
     model,
     optimizer,
     criterion,
     scheduler) = create(device)

    # Run the training and testing
    for epoch in range(1, 31):
        # This function is called to perform training for one epoch. It takes the model, the training data loader, the
        # optimizer, and the current epoch as inputs. During training, the model's weights are updated to minimize the
        # loss, computed using the specified optimizer and loss function (criterion).
        train(model, device, train_loader, optimizer, epoch, criterion)
        # This function is called to perform testing (evaluation) for one epoch. It evaluates the model's performance on
        # the test dataset without updating the model's weights. It calculates and prints the test loss and accuracy.
        test(model, device, test_loader, criterion)
        # This line is executed after each epoch. It's responsible for updating the learning rate of the optimizer.
        # Learning rate scheduling is a technique used to adjust the learning rate during training. The specific
        # scheduler used here is not shown in the provided code snippet, but it's typically created and defined earlier
        # in the script. The scheduler's role is to dynamically adjust the learning rate, which can help improve
        # training convergence.
        scheduler.step()


if __name__ == '__main__':
    main()
