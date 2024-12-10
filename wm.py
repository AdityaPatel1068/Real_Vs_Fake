import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

# Define the CNNModel as per your architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input shape (3, 128, 128)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape (32, 64, 64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input shape (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape (64, 32, 32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Input shape (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output shape (128, 16, 16)
        )
        self.flatten = nn.Flatten()  # Flattens to shape (128 * 16 * 16)
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)  # Binary classification output (sigmoid activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Binary classification with sigmoid
        return x

# Load the PyTorch model
@st.cache_resource
def load_model():
    # Instantiate the model
    model = CNNModel()

    # Define the path to the state dictionary
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "CNN.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load the state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Standard normalization
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict_image(image):
    with torch.no_grad():
        outputs = model(image)
        return outputs.item()  

# Streamlit App
st.title("Fake or Real Image ?")
st.write("Upload an image to check whether it is fake or real.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3-channel RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Get prediction
    prediction = predict_image(processed_image)

    # Display the result
    if prediction > 0.5:  # Assuming a threshold of 0.5 for binary classification
        st.write("**Result: Fake Image 🔴**")
    else:
        st.write("**Result: Real Image 🟢**")
