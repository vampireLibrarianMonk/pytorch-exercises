import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to normalize the series
def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series, min_val, max_val


# Function to create the dataset
def create_dataset(input_data, past, future):
    X, y = [], []
    for i in range(len(input_data) - past - future + 1):
        X.append(input_data[i:(i + past)])
        y.append(input_data[(i + past):(i + past + future)])
    return np.array(X), np.array(y)


# LSTM model class
class LSTMForecastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMForecastModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Training function
def train_model(model, train_loader, valid_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.float(), targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        print(f'Epoch {epoch + 1}, Validation Loss: {valid_loss / len(valid_loader)}')


# Function to make predictions
def make_predictions(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.float()
            outputs = model(inputs)
            predictions.append(outputs.numpy())
    return np.concatenate(predictions)


if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')

    temperatures = df['Temp'].values
    normalized_temps, min_temp, max_temp = normalize_series(temperatures)

    n_past = 10
    n_future = 1
    batch_size = 32

    X, y = create_dataset(normalized_temps, n_past, n_future)
    X_train, y_train = X[:int(len(X) * 0.8)], y[:int(len(y) * 0.8)]
    X_valid, y_valid = X[int(len(X) * 0.8):], y[int(len(y) * 0.8):]

    train_data = TensorDataset(torch.from_numpy(X_train).unsqueeze(-1), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(X_valid).unsqueeze(-1), torch.from_numpy(y_valid))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    model = LSTMForecastModel(input_dim=1, hidden_dim=100, output_dim=n_future, num_layers=2)

    train_model(model, train_loader, valid_loader, num_epochs=20, learning_rate=0.001)

    predicted_temperatures = make_predictions(model, valid_loader)
    predicted_temperatures_rescaled = predicted_temperatures * (max_temp - min_temp) + min_temp
    y_valid_rescaled = y_valid * (max_temp - min_temp) + min_temp

    plt.figure(figsize=(15, 6))
    plt.plot(predicted_temperatures_rescaled.flatten(), label="Predicted Temperatures", color='red', linestyle='--')
    plt.plot(y_valid_rescaled.flatten(), label="Actual Temperatures", color='blue')
    plt.title("Comparison of Predicted and Actual Temperatures")
    plt.xlabel("Time Steps")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()

