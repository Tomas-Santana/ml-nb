import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

# EMNIST dataset for letters (A-Z) classification
# Data source: https://www.kaggle.com/datasets/crawford/emnist

CLASS_MAP = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

class EMNISTDataset(Dataset):
    def __init__(self, train_path, test_path):
        # Load the test and train data
        train_df = pd.read_csv(train_path)
        self.train_digits = torch.tensor(train_df.iloc[:, 1:].to_numpy() / 255.0).float().view(-1, 1, 28, 28)
        self.train_labels = torch.tensor(train_df.iloc[:, 0].to_numpy() - 1)

        test_df = pd.read_csv(test_path)
        self.test_digits = torch.tensor(test_df.iloc[:, 1:].to_numpy() / 255.0).float().view(-1, 1, 28, 28)
        self.test_labels = torch.tensor(test_df.iloc[:, 0].to_numpy() - 1)

        self.transform = transforms.Compose([
            transforms.RandomRotation(10),  # Rotate by up to 10 degrees
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
        ])

    def __getitem__(self, idx):
        digit = self.train_digits[idx]
        label = self.train_labels[idx]
        if self.transform:
            digit = self.transform(digit)
        return digit, label
    
    def __len__(self):
        return len(self.train_digits)
    

class ConvBlock(nn.Module):
    """ Convolutional Block with Conv2d, BatchNorm2d and ReLU activation """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class EMNISTCNN(nn.Module):
    """ Convolutional Neural Network for EMNIST dataset """
    def __init__(self):
        
        super(EMNISTCNN, self).__init__()
        self.conv1 = ConvBlock(1, 32, 5, 2)
        self.conv2 = ConvBlock(32, 64) 
        self.conv3 = ConvBlock(64, 128) 
        self.conv4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.03)
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 26)
    
    def forward(self, x):
        """ Four convolutional layers followed by two fully connected layers. Max pooling and dropout are applied after each convolutional layer. """
        x = self.conv1(x) # 32 x 28 x 28
        x = self.pool(x) # 32 x 14 x 14
        x = self.dropout(x)
        x = self.conv2(x) # 64 x 14 x 14
        x = self.pool(x) # 64 x 7 x 7
        x = self.dropout(x)
        x = self.conv3(x) # 128 x 7 x 7
        x = self.conv4(x) # 256 x 7 x 7
        x = self.pool(x) # 256 x 3 x 3
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
    
def train(epochs=10, batch_size=16, lr=1e-3, checkpoint_path=None):
    train_path = "datasets/emnist-letters/emnist-letters-train.csv"
    test_path = "datasets/emnist-letters/emnist-letters-test.csv"
    
    dataset = EMNISTDataset(train_path, test_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EMNISTCNN()
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
    
    # We will use CrossEntropyLoss for multi-class classification and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Schedule the learning rate to decay every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients, forward pass, backward pass and optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        # save model checkpoint
        torch.save(model.state_dict(), f"src/8/checkpoints/emnist_cnn_epoch_{epoch+1}.pth")

        # Evaluate on the test set every epoch
        with torch.no_grad():
            pred = torch.argmax(model(dataset.test_digits.to(device)), dim=1)
            acc = (pred == dataset.test_labels.to(device)).float().mean()
            print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train(epochs=10, batch_size=16, lr=1e-3)