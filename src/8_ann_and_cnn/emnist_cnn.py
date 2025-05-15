from pyexpat import model
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
            transforms.RandomRotation(15),  # Rotate by up to 10 degrees
            transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),  # Random translation
        ])

    def __getitem__(self, idx):
        digit = self.train_digits[idx]
        label = self.train_labels[idx]
        digit = self.transform(digit)
        return digit, label
    
    def __len__(self):
        return len(self.train_digits)
    
class EMNISTCNN(nn.Module):
    """ Convolutional Neural Network for EMNIST dataset """
    def __init__(self):
        
        super(EMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # 32 x 28 x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 x 28 x 28
        # Maxpool2d 64 x 14 x 14
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 16 x 14 x 14
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 16 x 14 x 14
        # Maxpool2d 16 x 6 x 6
        self.fc1 = nn.Linear(16*7*7 , 52)
        self.out = nn.Linear(52, 26)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.BatchNorm2d(64)(x)
        x = F.dropout(x, .1)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = nn.BatchNorm2d(32)(x)
        x = F.relu(self.conv4(x))
        x = nn.BatchNorm2d(16)(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.out(x)
        
    
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
        torch.save(model.state_dict(), f"src/8_ann_and_cnn/checkpoints/emnist_cnn_epoch_{epoch+1}.pth")

        # Evaluate on the test set every epoch
        with torch.no_grad():
            pred = torch.argmax(model(dataset.test_digits.to(device)), dim=1)
            acc = (pred == dataset.test_labels.to(device)).float().mean()
            print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # train(epochs=10, batch_size=16, lr=1e-3)
    from torchview import draw_graph

    model = EMNISTCNN()
    draw_graph(model, input_size=(1, 1, 28, 28), device="cpu")