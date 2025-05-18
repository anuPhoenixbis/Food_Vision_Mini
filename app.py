import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Load class names
class_names = ['pizza', 'steak', 'sushi']

# Load trained model
@st.cache_resource
def load_model():
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=3)
    )
    model.load_state_dict(torch.load("models/effnetb2.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Get transforms
transform = models.EfficientNet_B2_Weights.DEFAULT.transforms()

# UI
st.title("🍕🥩🍣 FoodVision Mini Classifier")
st.write("Upload an image of pizza, steak or sushi. The model will predict what it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        # Preprocess and predict
        input_tensor = transform(image).unsqueeze(0)
        with torch.inference_mode():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze()

        # Show results
        for i in range(len(class_names)):
            st.write(f"**{class_names[i].capitalize()}**: {probs[i].item():.4f}")
        
        st.success(f"Prediction: {class_names[probs.argmax()]}")

 