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
st.title("🍕🥩🍣 FoodVision Mini: What's on your plate?")
st.write("Upload an image of pizza, steak or sushi. The model will predict what it is.")

uploaded_file = st.file_uploader("Upload an image of pizza, steak, or sushi (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

with st.expander("About this App"):
    st.write("""
    This app uses a machine learning model to classify images of food items like pizza, steak, or sushi.
    It uses a pre-trained EfficientNet-B2 model.
    This is a mini version for demonstration purposes and works best with clear images of pizza, steak, or sushi. It may not be accurate for other food items or complex images.
    """)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Classifying..."):
            # Preprocess and predict
            input_tensor = transform(image).unsqueeze(0)
        with torch.inference_mode():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze()

        # Show results
        # Create a DataFrame for the bar chart
        import pandas as pd
        chart_data = pd.DataFrame(
            probs.numpy(),
            index=class_names
        )
        st.bar_chart(chart_data)

        # Display top prediction and confidence
        top_prob, top_cat_idx = torch.max(probs, dim=0)
        top_class_name = class_names[top_cat_idx]
        st.success(f"Predicted: {top_class_name.capitalize()} (Confidence: {top_prob.item():.4f})")

 