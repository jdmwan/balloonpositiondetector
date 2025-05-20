import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from BalloonNetCNNBOX import BalloonNetCNN
from BalloonBoxDataset import BalloonBBoxDataset
from iou import calculate_iou


# ✅ 2️⃣ Load & Preprocess Image Data
def load_data(batch_size=16, folder = "BalloonDataset/train", csv = "BalloonDataset/labels_train.csv"):
    transform = transforms.Compose([
        #  Apply transformations (Resize, ToTensor, Normalize)
        transforms.Resize((832,368)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225]),
        # some transforms to try
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomPerspective(distortion_scale=0.2),
    ])
    # train_dataset = ImageFolder(root=folder, transform=transform)
    dataset= BalloonBBoxDataset(csv_file=csv, image_dir=folder, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# ✅ 3️⃣ Define Training Loop
def train_model(model, train_loader, num_epochs=500, learning_rate=0.0005):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma= 0.1)
    # criterion = nn.MSELoss()  # old loss
    criterion = nn.SmoothL1Loss()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Move images & labels to device (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)
            scaling_vector = torch.tensor([1/8.02,1/8.15,1/8.02,1/8.15]).to(device)

            # Zero the gradients
            optimizer.zero_grad()
            #  Forward pass
            output = model(images)
            labels_scaled = labels*scaling_vector
            #  Compute loss
            loss = criterion(output, labels_scaled)
            #  Backward pass (gradient calculation)
            loss.backward()
            #  Update weights
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        scheduler.step()
        #stop to avoid overfitting
        if loss.item() == 0.02:
            return


# ✅ 4️⃣ Evaluate the Model
def test_model(model, test_loader, iou_threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    mse_total = 0.0
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            #scale to correct pixels
            scaling_vector = torch.tensor([8.02,8.15,8.02,8.15]).to(device)

            outputs = model(images)
            mse_total += criterion(outputs, labels).item()
            #calculation for accuracy
            for pred, true in zip(outputs, labels):
                pred_scaled = pred*scaling_vector
                iou = calculate_iou(pred_scaled, true)
                if iou >= iou_threshold:
                    correct += 1
                total += 1

    accuracy = correct / total * 100
    avg_loss = mse_total / len(test_loader)

    print(f"Test Accuracy (IoU ≥ {iou_threshold}): {accuracy:.2f}%")
    print(f"Avg MSE Loss: {avg_loss:.4f}")


# ✅ 5️⃣ Save & Load Model
def save_model(model, filename="balloon_pos.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


# ✅ 6️⃣ Run Everything
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_data(folder = "BalloonDataset/train", csv = "BalloonDataset/labels_train.csv")
    test_loader = load_data(folder = "BalloonDataset/test", csv = "BalloonDataset/labels_test.csv")  # You might want to split into train & test folders
    # model = BalloonNet()
    model = BalloonNetCNN()
    model.to(device)

    train_model(model, train_loader)
    test_model(model, test_loader)
    save_model(model)
