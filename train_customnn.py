import os.path
from argparse import ArgumentParser

import joblib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from models.CustomNN import OtitisMediaClassifier
from models.models import ModelZoo
from utils.io import read_as_csv
from utils.preprocessing import image_transforms, label_transforms


def train_neural_network(data_root, train_csv, test_csv, model, checkpoint_path, epochs=20, lr=0.01, batch_size=64):
    train_path = os.path.join(data_root, train_csv)
    train_files, train_labels = read_as_csv(train_path)

    # Apply transformations
    X_train = np.array(
        [image_transforms(file, label) for file, label in zip(train_files, train_labels)]
    )
    Y_train = np.array([label_transforms(lab) for lab in train_labels])

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long()

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    nn_model = OtitisMediaClassifier()
    optimizer = optim.SGD(nn_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = nn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch: {e+1} \t Training Loss: {running_loss / len(trainloader)}')

    # Save the model
    torch.save(nn_model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="train",
        description="Script to train model",
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as config_file:
        configs = yaml.safe_load(config_file)

    # Model instantiation based on the provided configuration
    model = ModelZoo(**configs["model"]).get_model()

    train_neural_network(
        data_root=configs["data_root"],
        train_csv=configs["train_csv"],
        test_csv=configs["test_csv"],
        model=model,
        checkpoint_path=configs["checkpoint_path"],
        epochs=configs.get("epochs", 20),
        lr=configs.get("lr", 0.01),
        batch_size=configs.get("batch_size", 64)
    )
