import torch
import torchvision.transforms as transforms
from BalloonNetCNNBOX import BalloonNetCNN
from PIL import Image

# Define the inference transforms (matching the training pipeline)
transform = transforms.Compose([
    transforms.Resize((832, 368)),  # Resize the image
    transforms.RandomHorizontalFlip(p=0.5),  # Apply randomly during inference (if needed)
    transforms.RandomRotation(30),  # Rotate randomly (if needed)
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize (ImageNet stats)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color jitter (for training)
    transforms.GaussianBlur(kernel_size=3),  # Apply Gaussian blur (for training)
    transforms.RandomPerspective(distortion_scale=0.2),  # Random perspective (for training)
])

if __name__ == "__main__":
    # Load the pre-trained model
    model = BalloonNetCNN()
    try:
        model.load_state_dict(torch.load("balloon_pos.pth"))
    except:
        print("load failed")
    model.eval()  # Set the model to evaluation mode
    test_image_link = "BalloonDataset/test/image_0402.png"
    print(test_image_link)
    # Open and preprocess the image
    test_image_raw = Image.open(test_image_link)
    test_image = transform(test_image_raw).unsqueeze(0)  # Add batch dimension

    # Move the image and model to the same device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = test_image.to(device)
    model = model.to(device)

    # Perform inference
    with torch.no_grad():  # Disable gradient calculation during inference
        output = model(input_image)

    # Scale the output by 8 (due to pooling layers)
    print(output * 8.02)  # Print the scaled output
