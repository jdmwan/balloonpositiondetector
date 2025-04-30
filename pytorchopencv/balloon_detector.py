import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from BalloonNetCNN import BalloonNetCNN

# ✅ 1️⃣ Define the Neural Network
class BalloonNet(nn.Module):
    def __init__(self):
        super(BalloonNet, self).__init__()
        self.height = 128
        self.length = 128
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(3*self.height*self.length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        # 🔹 Define layers here (use nn.Linear for fully connected layers)

    def forward(self, x):
        # 🔹 Implement forward pass (pass input through layers)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

# ✅ 2️⃣ Load & Preprocess Image Data
def load_data(batch_size=16, folder = "BalloonDataset/train"):
    transform = transforms.Compose([
        # 🔹 Apply transformations (Resize, ToTensor, Normalize)
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(root=folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# ✅ 3️⃣ Define Training Loop
def train_model(model, train_loader, num_epochs=5, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 🔹 Define loss function

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # 🔹 Move images & labels to device (CPU/GPU)
            # 🔹 Zero the gradients
            optimizer.zero_grad()
            # 🔹 Forward pass
            output = model(images)
            # 🔹 Compute loss
            loss = criterion(output, labels)
            # 🔹 Backward pass (gradient calculation)
            loss.backward()
            # 🔹 Update weights
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ✅ 4️⃣ Evaluate the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            
            # 🔹 Move images to device
            # 🔹 Forward pass
            output = model(images)

            # 🔹 Compare predictions with actual labels
            results = torch.argmax(output, dim = 1)
            
            for prediction, label in zip(results, labels):
                total += 1
                if prediction == label:
                    correct += 1
    print(f"Accuracy: {correct / total * 100:.2f}%")

# ✅ 5️⃣ Save & Load Model
def save_model(model, filename="balloon_detector.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename="balloon_detector.pth"):
    model = BalloonNet()  # recreate the model structure
    model.load_state_dict(torch.load(filename))
    model.eval()  # set to eval mode
    print(f"Model loaded from {filename}")
    return model

# ✅ 6️⃣ Run Everything
if __name__ == "__main__":
    train_loader = load_data()
    test_loader = load_data(folder = "BalloonDataset/test", batch_size=4)  # You might want to split into train & test folders
    # model = BalloonNet()
    model = BalloonNetCNN()

    train_model(model, train_loader)
    test_model(model, test_loader)
    save_model(model)
