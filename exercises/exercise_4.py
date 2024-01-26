# OrderedDict from collections: Provides a dictionary subclass that preserves the order in which keys were first added.
# This is useful for maintaining an ordered set of elements, especially when the order of elements matters for
# operations.
from collections import OrderedDict

# Counter from collections: A specialized dictionary subclass designed for counting hashable objects.
# It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values.
from collections import Counter

# load from json: A function to parse a JSON formatted string or file into a Python dictionary or list.
# It is commonly used for reading data from JSON files or parsing JSON responses from APIs.
from json import load as json_load

# train_test_split from sklearn.model_selection: A function used to split data arrays or matrices into random train
# and test subsets. This is crucial for evaluating the performance of machine learning models by training on one subset
# of the data and testing on another.
from sklearn.model_selection import train_test_split

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

# cuda from torch: Provides access to CUDA (Compute Unified Device Architecture) tensors and operations,
# allowing for tensor computations on NVIDIA GPUs for accelerated computing.
from torch import cuda as torch_cuda

# device from torch: A context manager or a device object used to specify the device (CPU or GPU) on which tensors
# are allocated and operations are performed. It's essential for directing computations to specific hardware.
from torch import device as torch_device

# The specific components imported are:
# 2. **device**: This function is utilized to set up the device on which to perform computations
# (e.g., 'cuda' or 'cpu').
from torch import device as torch_device

# float32 from torch: Specifies the data type for tensors as 32-bit floating points.
# This data type is commonly used for storing tensor values, especially when precision is balanced with memory
# efficiency.
from torch import float as torch_float32

# no_grad from torch: A context manager that disables gradient calculation, making code run faster and reducing memory
# usage by not keeping track of operations that require gradients. This is typically used during inference.
from torch import no_grad

# sigmoid from torch: Provides the Sigmoid activation function, which is widely used in binary classification problems
# and neural network layers for introducing non-linearity.
from torch import sigmoid

# long from torch: Specifies the data type for tensors as long integers (64-bit). This is often used for tensors
# that store indices or counts.
from torch import long as torch_long

# tensor from torch: A function to create a tensor object in PyTorch, which is a multi-dimensional matrix containing
# elements of a single data type. Tensors are fundamental to PyTorch operations and models.
from torch import tensor as torch_tensor

# __version__ as torch_version**: This imports the version identifier of the PyTorch library (aliased as
#    'torch_version'). It's helpful for compatibility checks or when reporting issues.
from torch import version as torch_func_version, __version__ as torch_version

# zeros from torch: A function to create a tensor filled with zeros. This is useful for initializing weights or
# creating tensors with a specific shape as placeholders.
from torch import zeros as torch_zeroes

# nn from torch: Provides the building blocks for creating neural networks, including layers, activation functions,
# and loss functions. It's a foundational module for defining and assembling neural network architectures.
import torch.nn as nn

# nn.functional from torch: Contains functions like activation functions, loss functions, and convolution operations.
# These functions are stateless, providing a functional interface to operations that can be applied to tensors.
import torch.nn.functional as F

# optim from torch: Implements various optimization algorithms for training neural networks, including SGD, Adam,
# and RMSprop. These algorithms are used to update the weights of the network during training.
import torch.optim as optim

# DataLoader and Dataset from torch.utils.data: DataLoader provides an iterable over a dataset, with support for
# batching, sampling, shuffling, and multiprocess data loading. Dataset is an abstract class for representing
# a dataset, with custom implementations required to define how data is loaded and processed.
from torch.utils.data import DataLoader, Dataset

# get_tokenizer from torchtext.data.utils: A utility function to obtain a tokenizer. Tokenizers are used to convert
# text into a sequence of tokens or words, which is a common preprocessing step for text data in natural language
# processing.
from torchtext.data.utils import get_tokenizer

# vocab from torchtext.vocab: A class for building a vocabulary from a set of tokens. This vocabulary can then map
# tokens to indices, which is a common requirement for converting text data into a numerical form that can be processed
# by models.
from torchtext.vocab import vocab as torch_text_vocab

# urlretrieve from urllib.request: A utility function to download a file from a given URL. This is useful for
# downloading data sets, pre-trained models, or other resources required by a project from the internet.
from urllib.request import urlretrieve


# Define a class named SarcasmDataset, which is a subclass of PyTorch's Dataset class.
# This custom dataset class is used to handle sarcasm data (sentences and labels).

# In Python, the use of double underscores (also known as "dunder" or "magic" methods) before and after the names of
# certain methods, serves a specific purpose. These methods are part of Python's protocol for built-in behaviors and
# are not meant to be called directly by the user, but rather by Python itself to perform certain operations.
class SarcasmDataset(Dataset):

    # The constructor method initializes the dataset object with sentences and their corresponding labels.
    def __init__(self, sentences, labels):
        self.sentences = sentences  # Store the provided sentences in the instance variable.
        self.labels = labels  # Store the provided labels in the instance variable.

    # The __len__ method returns the number of items in the dataset.
    # It's a required method for the Dataset class, allowing PyTorch to know the dataset size.
    def __len__(self):
        return len(self.sentences)  # Return the total number of sentences.

    # The __getitem__ method retrieves a single data point from the dataset.
    # It's a required method for the Dataset class, enabling indexing and iteration over the dataset.
    def __getitem__(self, idx):
        # Return the sentence and its corresponding label at the specified index.
        # This allows for direct access to a data sample and its label using dataset[index].
        return self.sentences[idx], self.labels[idx]


# Define a function named create_vocab that takes a list of sentences and a maximum vocabulary size as input.
def create_vocab(sentences, max_size):
    # Get a tokenizer function for basic English from torchtext's utility functions. This tokenizer splits sentences
    # into tokens (words).
    tokenizer = get_tokenizer("basic_english")

    # Initialize a Counter object from the collections module to count occurrences of each token.
    counter = Counter()

    # Iterate over each sentence in the provided list of sentences.
    for sentence in sentences:
        # Update the counter with tokens from the current sentence, incrementing counts for each token found by the
        # tokenizer.
        counter.update(tokenizer(sentence))

    # Retrieve the most common tokens up to the limit of max_size minus 2, to leave space for special tokens.
    # This ensures the final vocabulary size doesn't exceed max_size.
    most_common_tokens = counter.most_common(max_size - 2)  # Adjust for special tokens

    # Create an OrderedDict and initialize it with two special tokens: '<pad>' for padding and '<unk>' for unknown
    # tokens, assigning them indices 0 and 1, respectively.
    extended_vocab = OrderedDict([('<pad>', 0), ('<unk>', 1)])

    # Update the OrderedDict with the most common tokens, which will be assigned subsequent indices starting from 2.
    extended_vocab.update(most_common_tokens)

    # Create a Vocab object from the extended vocabulary containing both the special tokens and the most common tokens.
    # The Vocab object provides a mapping from tokens to indices and is used for numericalizing text data.
    return torch_text_vocab(extended_vocab)


# Define the function encode_sentences, which takes a list of sentences, a vocabulary, and a tokenizer as input
# parameters.
def encode_sentences(sentences, vocab, tokenizer):
    # Initialize an empty list to hold the encoded versions of each sentence.
    encoded_sentences = []

    # Define a special token '<unk>' that represents any words not found in the vocabulary.
    unk_token = '<unk>'

    # Retrieve the index assigned to the unknown token '<unk>' in the provided vocabulary.
    # This index will be used for words that are not present in the vocabulary.
    unk_idx = vocab[unk_token]

    # Iterate over each sentence in the list of sentences provided to the function.
    for sentence in sentences:
        # Use the provided tokenizer function to split the sentence into individual tokens (words).
        tokenized_sentence = tokenizer(sentence)

        # For each token in the tokenized sentence, check if the token exists in the vocabulary.
        # If the token exists, use its corresponding index from the vocabulary. If not, use the index for '<unk>'.
        # This results in a list of indices that represent the encoded sentence.
        encoded_sentence = [vocab[token] if token in vocab else unk_idx for token in tokenized_sentence]

        # Add the encoded sentence (a list of indices) to the list of encoded sentences.
        encoded_sentences.append(encoded_sentence)

    # After processing all sentences, return the list of encoded sentences.
    return encoded_sentences


# Define the function pad_sequences, which takes a list of numerical sequences and a maximum sequence length as input
# parameters.
def pad_sequences(sequences, max_length):
    # Create a tensor of zeros with the shape (number of sequences, max_length). This tensor will hold the padded
    # sequences.
    # The dtype=torch_long specifies that the elements of the tensor are long integers, suitable for holding indices.
    padded_sequences = torch_zeroes((len(sequences), max_length), dtype=torch_long)

    # Enumerate over the list of sequences to get both the index (i) and the sequence itself (seq) for each sequence in
    # the list.
    for i, seq in enumerate(sequences):
        # Determine the length to which the current sequence should be padded. This is the smaller of max_length or the
        # sequence's actual length.
        # This ensures that sequences longer than max_length are truncated to max_length.
        length = min(max_length, len(seq))

        # Update the ith row of padded_sequences to include the first 'length' elements of the current sequence.
        # torch_tensor(seq[:length], dtype=torch_long) creates a tensor from the first 'length' elements of seq,
        # ensuring that the data type is still long integer. The tensor is then assigned to the first 'length' positions
        # in the ith row of padded_sequences, effectively padding the sequence with zeros if it is shorter than
        # max_length.
        padded_sequences[i, :length] = torch_tensor(seq[:length], dtype=torch_long)

    # After processing all sequences, return the tensor containing all the padded sequences.
    return padded_sequences


# Define a class named SarcasmModel, which inherits from nn.Module. This class represents a neural network model.
class SarcasmModel(nn.Module):
    # The constructor method initializes the neural network with layers specified by the given parameters.
    def __init__(self, vocab_size, embedding_dim, output_dim, max_length):
        # Initialize the base class (nn.Module).
        super(SarcasmModel, self).__init__()

        # Embedding layer that converts token indices to dense vectors of a specified size (embedding_dim).
        # The 'padding_idx=0' argument ensures that the padding token (index 0) maps to a zero vector.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Dropout layer for regularization, randomly zeroes some of the elements of the input tensor with probability
        # 0.2.
        self.dropout = nn.Dropout(0.2)

        # 1D Convolutional layer that applies 32 filters of size 5 to the input embeddings.
        self.conv1d = nn.Conv1d(embedding_dim, 32, kernel_size=5)

        # Max pooling layer that applies a 1D max pooling over an input signal composed of several input planes, using a
        # window of size 4.
        self.maxpool = nn.MaxPool1d(kernel_size=4)

        # LSTM layer for processing sequences, with 32 input features and 64 hidden units, bidirectional processing.
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)

        # Adaptive average pooling layer to convert the output of LSTM to a fixed size output, facilitating connection
        # to dense layers.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer that maps the LSTM output to a vector of size 128.
        self.fc1 = nn.Linear(128, 128)

        # Second fully connected layer that maps the 128-sized vector to the final output size.
        self.fc2 = nn.Linear(128, output_dim)

    # The forward method defines the forward pass of the input through the model.
    def forward(self, x):
        # Pass the input x through the embedding layer.
        x = self.embedding(x)

        # Permute the dimensions of x to match the expected input shape of Conv1D (batch_size, channels,
        # sequence_length).
        x = x.permute(0, 2, 1)

        # Apply dropout for regularization.
        x = self.dropout(x)

        # Apply the ReLU activation function to the output of the convolutional layer.
        x = F.relu(self.conv1d(x))

        # Apply max pooling to reduce the dimensionality and to extract the most significant features.
        x = self.maxpool(x)

        # Permute the dimensions back to match the expected input shape of LSTM (batch_size, sequence_length, channels).
        x = x.permute(0, 2, 1)

        # Pass the result through the LSTM layer. Only the output tensor is used, and the hidden state is ignored.
        x, _ = self.lstm(x)

        # Permute again for the adaptive average pooling layer.
        x = x.permute(0, 2, 1)

        # Apply adaptive average pooling and remove the unnecessary extra dimension.
        x = self.global_avg_pool(x).squeeze(2)

        # Apply ReLU activation function to the output of the first fully connected layer.
        x = F.relu(self.fc1(x))

        # Apply dropout again.
        x = self.dropout(x)

        # Apply the sigmoid activation function to the output of the second fully connected layer to get the final model
        # output.
        x = sigmoid(self.fc2(x))

        # Return the final output.
        return x


# Define the train_model function with parameters for the model, data loaders for training and validation,
# the device to run the training on (CPU or GPU), and the number of epochs to train for.
def train_model(model, train_loader, val_loader, device, epochs=10):
    # Define the loss function to be Binary Cross-Entropy, suitable for binary classification tasks.
    criterion = nn.BCELoss()

    # Initialize the optimizer to use the Adam algorithm for optimizing the model parameters.
    optimizer = optim.Adam(model.parameters())

    # Transfer the model to the specified device (CPU or GPU).
    model.to(device)

    # Loop over the dataset multiple times, each loop is an epoch.
    for epoch in range(epochs):
        # Set the model to training mode. This is necessary because certain layers like Dropout behave differently
        # during training.
        model.train()

        # Initialize a variable to keep track of the accumulated loss over the epoch.
        running_loss = 0.0

        # Iterate over the training data (inputs and labels).
        for inputs, labels in train_loader:
            # Transfer the inputs and labels to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients to prevent accumulation from previous iterations.
            optimizer.zero_grad()

            # Forward pass: compute the predicted outputs by passing inputs to the model and squeeze the output if
            # necessary.
            outputs = model(inputs).squeeze(1)

            # Compute the loss between the predicted outputs and the actual labels.
            loss = criterion(outputs, labels.float())

            # Backward pass: compute the gradient of the loss with respect to the model parameters.
            loss.backward()

            # Perform a single optimization step (parameter update).
            optimizer.step()

            # Accumulate the loss over the batch to later calculate the average loss for the epoch.
            running_loss += loss.item()

        # Print the average loss for the epoch.
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        # Switch the model to evaluation mode. This is necessary because certain layers like Dropout behave differently
        # during evaluation.
        model.eval()

        # Initialize counters for the total number of labels and the number of correct predictions.
        total = 0
        correct = 0

        # Disable gradient computation since it's not needed for validation, which saves memory and computations.
        with no_grad():
            # Iterate over the validation data.
            for inputs, labels in val_loader:
                # Transfer the inputs and labels to the specified device.
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass: compute the predicted outputs by passing inputs to the model and squeeze the output if
                # necessary.
                outputs = model(inputs).squeeze(1)

                # Apply a threshold of 0.5 to the outputs to obtain the binary predictions.
                predicted = outputs.round()

                # Update the total count of labels processed.
                total += labels.size(0)

                # Update the count of correct predictions.
                correct += (predicted == labels).sum().item()

        # Print the accuracy for the current epoch, calculated as the percentage of correct predictions.
        print(f'Accuracy after epoch {epoch + 1}: {100 * correct / total}%')


# Define the function create_model with a parameter for specifying the computation device (e.g., CPU or GPU).
def create_model(input_device):
    # URL pointing to the sarcasm dataset.
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'

    # Download the dataset and save it locally as 'sarcasm.json'.
    urlretrieve(url, 'sarcasm.json')

    # Initialize lists to hold sentences and their corresponding sarcasm labels.
    sentences = []
    labels = []

    # Open the downloaded JSON file for reading.
    with open('sarcasm.json', 'r') as file:
        # Load the JSON content.
        data = json_load(file)

        # Iterate over each record in the dataset, extracting the sentence and its sarcasm label.
        for item in data:
            sentences.append(item['headline'])  # Add the sentence to the list.
            labels.append(item['is_sarcastic'])  # Add the sarcasm label (0 or 1) to the list.

    # Set key parameters for the model and data processing.
    vocab_size = 1000
    max_length = 120
    embedding_dim = 100
    output_dim = 1

    # Obtain a tokenizer for basic English to convert sentences into tokens.
    tokenizer = get_tokenizer("basic_english")

    # Create a vocabulary from the list of sentences with the specified maximum size.
    vocab = create_vocab(sentences, vocab_size)

    # Encode the sentences into sequences of indices based on the vocabulary.
    encoded_sentences = encode_sentences(sentences, vocab, tokenizer)

    # Pad the encoded sequences to ensure they all have the same length.
    padded_sentences = pad_sequences(encoded_sentences, max_length)

    # Convert the labels list into a tensor of type float32.
    labels = torch_tensor(labels, dtype=torch_float32)

    # Split the padded sentences and labels into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(padded_sentences, labels, test_size=0.1, random_state=42)

    # Create dataset objects for training and testing data.
    train_data = SarcasmDataset(x_train, y_train)
    test_data = SarcasmDataset(x_test, y_test)

    # Initialize DataLoader objects for batching and shuffling the training and testing datasets.
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Initialize the sarcasm detection model with the specified parameters.
    model = SarcasmModel(len(vocab), embedding_dim, output_dim, max_length)

    # Train the model using the training and validation DataLoader objects, on the specified device.
    train_model(model, train_loader, val_loader, input_device)

    # Return the trained model.
    return model


if __name__ == '__main__':
    print("Software Versions:")

    # CUDA
    if torch_cuda.is_available():
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

    print("\n")

    created_model = create_model(device)
    # torch.save(created_model.state_dict(), "../models/exercise_2_pytorch.pth")
