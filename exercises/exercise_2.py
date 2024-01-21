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
# 3. **flatten**: A function to flatten a tensor. It collapses a multi-dimensional tensor into a single dimension,
#    often used in transitioning from convolutional layers to fully connected layers in a neural network.
# 4. **no_grad**: This context manager is critical for inference or validation phases, where you do not want operations
#    to track gradients. It reduces memory usage and speeds up computations.
# 5. **__version__ as torch_version**: This imports the version identifier of the PyTorch library (aliased as
#    'torch_version'). It's helpful for compatibility checks or when reporting issues.
from torch import cuda, device, flatten, no_grad, version as torch_func_version, __version__ as torch_version

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

# This line imports a package in the PyTorch library that consists of popular datasets, model architectures, and common
# image transformations for computer vision. This line imports two submodules: datasets for accessing various standard
# datasets and transforms for performing data preprocessing and augmentation operations on images.
from torchvision import datasets, transforms, __version__ as torchvision_version

# This imports the StepLR class from the lr_scheduler submodule in optim. Learning rate schedulers adjust the learning
# rate during training, which can lead to more effective and faster training. StepLR decreases the learning rate at
# specific intervals, which is helpful in fine-tuning the network as training progresses.
from torch.optim.lr_scheduler import StepLR

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

# TorchVision
print("\tTorchvision:", torchvision_version)

print("\n")


# Define the neural network architecture
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes=10):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3 * 3 * 32, num_classes)  # Updated based on the final feature map size

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.bn3(x)
        x = nn.functional.max_pool2d(x, 2)

        x = flatten(x, 1)
        x = self.dropout2(x)
        x = self.fc1(x)
        output = nn.functional.softmax(x, dim=1)
        return output


# Load the FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('../data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
model = FashionMNISTModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Initialize the StepLR scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)


# Training loop
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Testing loop
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Run the training and testing
for epoch in range(1, 31):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()  # Update the learning rate
