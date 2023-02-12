import torch 
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import argparse

# Define the Network Architecture
class Net(nn.Module):
    def __init__(self):
        # Initialize the nn.Module
        super().__init__()
        # Input of 3 channels, first convolutional layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional Layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Fully Connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 classes in the output
    
    def forward(self, x):
        ''' Forward pass through the network '''
        y_hat = self.pool(F.relu(self.conv1(x)))
        y_hat = self.pool(F.relu(self.conv2(y_hat)))
        y_hat = torch.flatten(y_hat, 1)
        y_hat = F.relu(self.fc1(y_hat))
        y_hat = F.relu(self.fc2(y_hat))
        y_hat = self.fc3(y_hat)

        return y_hat

# Test and Train Functions
def train(model: Net, epoch: int, loss_fn, opt, data_loader, verbose: bool):
    ''' A single training step '''
    model.train()

    epoch_loss = 0.0
    for batch_idx, (X, y) in enumerate(data_loader):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

        # Print loss every 3000 minibatches
        if batch_idx % 3000 == 2999 and verbose:
            print(f"Epoch: {epoch+1}, Minibatch: {batch_idx+1} : Loss = {(epoch_loss/3000):.5f}")

def test(model, loss_fn, data_loader, verbose: bool):
    ''' A single validation step '''
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for (X, y) in data_loader:
            pred = model(X)
            _, y_hat = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (y_hat == y).sum().item()

    if verbose:
        print(f"Accuracy: {100*correct // total}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Cifar10",
        description="Training a Convolutional Neural Network on the Cifar10 Dataset",
    )
    
    # Command Line Arguments

    parser.add_argument("--epochs", "--e", type=int, help="Number of epochs to train for. Default=15", default=15)
    parser.add_argument("--verbose", "--v", help="Print warnings and model training progress", action='store_true')
    parser.add_argument("--train_path", type=str, help="The path to the directory in which the Training Dataset will be stored. Default=(./data/)", default="./data/")
    parser.add_argument("--test_path", type=str, help="The path to the directory in which the Testing Dataset will be stored. Default=(./data/)", default="./data/")
    parser.add_argument("--checkpoints", type=str, help="The path to which model checkpoints will be saved. Default=(./models/)", default="./models/")
    parser.add_argument("--train_batch", type=int, help="The training batch size", required=True)
    parser.add_argument("--test_batch", type=int, help="The testing batch size", required=True)
    parser.add_argument("--learning_rate", "--lr", type=float, help="The Learning rate used by the Optimizer. Default=0.001", default=0.001, dest='lr')
    parser.add_argument("--momentum", "--m", type=float, help="The momentum used by SGD (if used). Default=0.9", default=0.9)

    args = parser.parse_args()

    # Print warnings and model training progress
    verbose = args.verbose.lower() == "y"

    # Get device to train model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and verbose:
        print("Warning: Training model on CPU.")

    # Load datasets

    # Transformation to normalize the dataset
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = CIFAR10(
        root=args.train_path,
        train=True,
        download=True,
        transform=trans
    )    

    test_set = CIFAR10(
        root=args.test_path,
        train=False,
        download=True,
        transform=trans
    )

    # Create DataLoaders

    training_loader = DataLoader(
        dataset=train_set,
        batch_size=args.training_batch,
        shuffle=True
    )

    testing_loader = DataLoader(
        dataset=test_set,
        batch_size=args.testing_batch,
        shuffle=True
    )

    # Create model
    model = Net().to(device)

    # Define our Loss Function and Optimizer    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_function = nn.CrossEntropyLoss()

    # Train the Network
    test(model, loss_function, testing_loader, verbose) # random network
    for epoch in range(args.epochs+1):
        train(model, epoch, loss_function, optimizer, training_loader, verbose)
        test(model, loss_function, testing_loader, verbose) 