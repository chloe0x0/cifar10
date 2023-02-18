import torch 
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import argparse
from train import Net
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Testing a Convolutional Neural Network on CIFAR10",
        description="Visualizes the performance of a Convolutional Neural Network on CIFAR10"
    )

    parser.add_argument("--model", type=str, help="The path to the model")
    parser.add_argument("--test_path", '--tp', type=str, help="The path to the testing set from which to take images from", dest='tp', default='./data/')
    parser.add_argument("--batch", type=int, help="The batch size (default: 8)", default=8)
    parser.add_argument("--confusion", '--cm', help="Whether or not to display a Confusion matrix for the model. default: False", action='store_true', dest='cm')

    args = parser.parse_args()

    # Get device to put model and Tensors on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: Could not find CUDA, evaluating model on CPU.")

    # Transformation used to normalize the images
    trans = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Get the test dataset
    test_set = CIFAR10(
        root=args.tp,
        train=False, 
        download=False,
        transform=torchvision.transforms.ToTensor()
    )

    # Data Loader
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch,
        pin_memory=device=="cuda",
        shuffle=True
    )

    # Load model
    model = Net().to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    # Get random images
    imgs, labels = next(iter(test_loader))

    # Get predictions
    X, y = trans(imgs), labels
    X, y = X.to(device), y.to(device)
    y_hat = model(X)
    y_hat = y_hat.data.max(1, keepdim=True)[1]

    to_img = torchvision.transforms.ToPILImage()

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    from_ord = {ix:k for ix,k in enumerate(classes)}

    # grid = torchvision.utils.make_grid(imgs)
    fig = plt.figure(figsize=(2, 4))
    for i in range(0, args.batch):
        ax = fig.add_subplot(2, 4, i+1, xticks=[], yticks=[])
        ax.axis('off')
        # get img from batch tensor
        img = imgs[i]
        img = to_img(img)
        ax.imshow(img)
        # get predicted and ground truth labels
        real = labels[i].item()
        pred = y_hat[i].item()
        col = "green" if pred==real else "red"
        ax.set_title(f"{from_ord[pred]}", color=col)    

    plt.show()

    # Build and display the Confusion Matrix
    if args.cm:
        true, pred = [], []

        for imgs, labels in test_loader:
            imgs = trans(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            y_hat = model(imgs)

            y_hat = (torch.max(torch.exp(y_hat), 1)[1]).data.cpu().numpy()
            pred.extend(y_hat)
            true.extend(labels.cpu().numpy())

        cm = confusion_matrix(true, pred)
        df = pd.DataFrame(
            cm/np.sum(cm, axis=1),
            index=[i for i in classes],
            columns = [i for i in classes]
        )

        plt.figure(figsize=(12,7))
        sn.heatmap(df, annot=True)
        plt.show()