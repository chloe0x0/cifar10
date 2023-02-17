import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import argparse

# Define the Network Architecture
class Net(nn.Module):
    def __init__(self):
        # Initialize the nn.Module
        super().__init__()
        # Convolutional Layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Fully Connected Layers
        self.lin = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        ''' Forward pass through the network '''
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.lin(x)

        return x

# Test and Train Functions
def train(model: Net, epoch: int, loss_fn, opt, data_loader, device, verbose: bool):
    ''' A single training step '''
    model.train()

    epoch_loss = 0.0
    for batch_idx, (X, y) in enumerate(data_loader):
        # Put feature and label tensors on the device
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()

        # Print loss every 3000 minibatches
        if batch_idx % 3000 == 2999 and verbose:
            print(f"Epoch: {epoch+1}, Minibatch: {batch_idx+1} : Loss = {(epoch_loss/3000):.5f}")
            epoch_loss = 0.0

def test(model, data_loader, device, verbose: bool):
    ''' A single validation step '''
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for (X, y) in data_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            _, y_hat = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (y_hat == y).sum().item()

    if verbose:
        print(f"Accuracy: {100*correct // total}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Cifar10",
        description="Training a Convolutional Neural Network on the Cifar10 Dataset",
    )
    
    # Command Line Arguments

    parser.add_argument("--epochs", "--e", type=int, help="Number of epochs to train for. Default=15", default=15, dest="epochs")
    parser.add_argument("--verbose", "--v", help="Print warnings and model training progress", action='store_true', dest="verbose")
    parser.add_argument("--train_path", type=str, help="The path to the directory in which the Training Dataset will be stored. Default=(./data/)", default="./data/")
    parser.add_argument("--test_path", type=str, help="The path to the directory in which the Testing Dataset will be stored. Default=(./data/)", default="./data/")
    parser.add_argument("--checkpoints", type=str, help="The path to which model checkpoints will be saved. Default=(./models/)", default="./models/trained_model.pt")
    parser.add_argument("--train_batch", type=int, help="The training batch size", default=8)
    parser.add_argument("--test_batch", type=int, help="The testing batch size", default=8)
    parser.add_argument("--learning_rate", "--lr", type=float, help="The Learning rate used by the Optimizer. Default=0.001", default=0.001, dest='lr')
    parser.add_argument("--momentum", "--m", type=float, help="The momentum used by SGD/Adam (if used). Default=0.9", default=0.9, dest="momentum")
    parser.add_argument("--serialize", "--s", help="Whether or not to save the model to the --checkpoints directory.", action='store_true', dest="serialize")

    args = parser.parse_args()

    # Get device to train model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.verbose:
        print("Warning: Could not find CUDA, training model on CPU.")

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

    use_pin_memory = device.type == "cuda"

    training_loader = DataLoader(
        dataset=train_set,
        batch_size=args.train_batch,
        shuffle=True,
        pin_memory=use_pin_memory
    )

    testing_loader = DataLoader(
        dataset=test_set,
        batch_size=args.test_batch,
        shuffle=True,
        pin_memory=use_pin_memory
    )

    # Create model
    model = Net().to(device)

    # Define our Loss Function and Optimizer    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_function = nn.CrossEntropyLoss()

    # Train the Network
    if args.verbose:
        print("Training model")
    test(model, testing_loader, device, args.verbose) # random network
    for epoch in range(args.epochs):
        train(model, epoch, loss_function, optimizer, training_loader, device, args.verbose)
        test(model, testing_loader, device, args.verbose) 

    # Serialize the trained model
    if args.serialize:
        if args.verbose: print(f"Saving trained model to {args.checkpoints}")
        torch.save(model.state_dict(), args.checkpoints)