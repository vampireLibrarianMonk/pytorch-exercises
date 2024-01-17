import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_model(model, criterion, optimizer, train_loader, patience=5, n_epochs=35):
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def create_dataset():
    depth = 10000
    training_range = np.arange(-depth, depth)
    offset = 7
    test_range = training_range + offset

    x_train = torch.tensor(training_range.reshape(-1, 1), dtype=torch.float32)
    y_train = torch.tensor(test_range.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader

def main():
    train_loader = create_dataset()
    model = SimpleLinearModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_model(model, criterion, optimizer, train_loader)

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Prediction
    predicted_depth = 100
    base_x = np.arange(-predicted_depth, predicted_depth + 1, 10)
    new_x_values = torch.tensor(base_x.reshape(-1, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predicted_y = model(new_x_values)
    
    print("Predicted y for x =", base_x, ":", predicted_y.numpy().flatten())

    # Save the final model
    torch.save(model.state_dict(), "exercise_1_model.pth")

if __name__ == '__main__':
    main()

