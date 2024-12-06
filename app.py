import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import torch.nn as nn

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and preprocess image
def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        return image
    except UnidentifiedImageError:
        return None

# Prediction function
def predict(model_path, img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()
        label = "Sick" if prediction > 0.5 else "Healthy"
    return label, prediction

# Function to check if the image resembles an X-ray
def is_xray(image):
    # Convert to grayscale
    grayscale_image = image.convert("L")  # Grayscale conversion
    # Compare pixel intensity differences
    difference = sum(
        abs(sum(rgb) / 3 - gray)  # Average RGB value vs grayscale value
        for rgb, gray in zip(image.getdata(), grayscale_image.getdata())
    )
    threshold = 1e5  # Adjust based on testing
    return difference < threshold

# Streamlit UI
st.set_page_config(page_title="Chest X-Ray Diagnostic Tool", page_icon="🩺")
st.title("AI for Healthcare: Chest X-Ray Diagnosis ")
st.markdown("Upload an X-ray image to check if the person is healthy or sick.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = load_image(uploaded_file)
    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Check if the image resembles an X-ray
        if not is_xray(image):
            st.error("The uploaded image does not appear to be an X-ray. Please upload a valid X-ray image.")
        else:
            # Predict button
            if st.button("Predict"):
                with st.spinner("Analyzing the image..."):
                    model_path = "xray_model.pth"  # Path to your saved model
                    label, confidence = predict(model_path, image)

                    # Display result
                    st.success(f"Prediction: **{label}**")
                    st.info(f"Confidence: {confidence:.4f}")
    else:
        st.error("Invalid file. Please upload an image file.")
