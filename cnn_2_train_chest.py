import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation and Test transform
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Updated dataset paths based on your directory structure:
# Directory: D:\Surgical AI assitant\brain\Chest
train_dataset = datasets.ImageFolder(root=r'D:\Surgical AI assitant\brain\Chest\train', transform=train_transform)
val_dataset   = datasets.ImageFolder(root=r'D:\Surgical AI assitant\brain\Chest\val',   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=r'D:\Surgical AI assitant\brain\Chest\test',  transform=val_test_transform)

# Print class-to-index mapping (should output something like ["Normal", "Pneumonia"])
class_names = train_dataset.classes
print("Class Labels:", class_names)

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN Model for Chest Classification (2 classes)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):  # For chest dataset, we have 2 classes: Normal and Pneumonia
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # The input to fc1 must match the output size after convolution & pooling.
        # Here, it is assumed to be 128 * 28 * 28 (you might need to adjust this if your images have different dimensions).
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Updated for 2 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model for 2 classes (Normal and Pneumonia)
num_classes = len(class_names)
model = CustomCNN(num_classes=num_classes).to(device)

# Hyperparameters
learning_rate = 0.001
epochs = 50
patience = 5  # Early stopping patience

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
best_val_loss = float('inf')
early_stop_count = 0
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}...")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc='Training', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
          f'Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), 'custom_cnn_2_chest.pth')  # Save best model for chest classification
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("Early stopping triggered.")
            break

# Testing the model
model.load_state_dict(torch.load('custom_cnn_2_chest.pth'))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.4f}')
