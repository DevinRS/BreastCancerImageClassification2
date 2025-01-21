import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# -- Helper Functions --
import cv2
from PIL import Image
# Function to convert PIL Image to OpenCV format
def pil_to_cv2(pil_image):
    # Convert PIL Image to RGB (to ensure compatibility)
    rgb_image = pil_image.convert("RGB")
    # Convert RGB to OpenCV BGR format
    open_cv_image = np.array(rgb_image)
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

# function: canny_count(img_path: str)
# Given a path to an image, count the amount of cell nuclei using canny edge detection
def canny_count(pil_image, blur_kernel_size=(11,11), low_threshold=30, high_threshold=105, morph_shape=cv2.MORPH_RECT, dilation_kernel_size=(1, 1), dilation_iterations=2, show_img=False):
    image = pil_to_cv2(pil_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # Conversion of image Into Gray Color
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(binary, blur_kernel_size, 0)    # Add Gaussian Blur
    canny = cv2.Canny(blur, low_threshold, high_threshold, 3)    # Using Canny Edge Detection Algorithm
    dilated = cv2.dilate(canny, cv2.getStructuringElement(morph_shape, dilation_kernel_size), iterations=dilation_iterations)    # Dilation is used to observe changes
    (cnt, heirarchy ) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)

    if show_img:
        return (len(cnt), gray, blur, canny, dilated, rgb)

    return len(cnt)

def process_image(pil_image, show_img=False):
    image = pil_to_cv2(pil_image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: Invert the mask if brown regions are darker
    binary_mask = cv2.bitwise_not(binary_mask)

    # Postprocessing: Remove small noise (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Calculate areas
    stained_area = cv2.countNonZero(binary_mask)
    total_area = binary_mask.size
    background_area = total_area - stained_area

    if show_img:
        return (stained_area / total_area, gray, binary_mask)

    # Return results
    return (stained_area / total_area)

import torch
import torch.nn as nn
import torchvision.models as models

class ConvNextWithCellCount(nn.Module):
    def __init__(self, base_model, cell_count_dim=1, num_classes=2):
        super(ConvNextWithCellCount, self).__init__()
        self.base_model = base_model
        self.cell_count_dim = cell_count_dim

        # Ensure the base model outputs features
        self.base_model.classifier = nn.Sequential(
            nn.LayerNorm([768, 1, 1], eps=1e-06, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        # Combine image features and cell_count
        self.fc = nn.Sequential(
            nn.Linear(768 + cell_count_dim, 512),  # Combine image features with cell_count
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x, cell_count):
        x = self.base_model(x)  # Extract image features
        cell_count = cell_count.unsqueeze(1)  # Ensure cell_count has the correct shape (batch_size, 1)
        x = torch.cat((x, cell_count), dim=1)  # Concatenate features
        x = self.fc(x)  # Pass through fully connected layers
        return x

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode   
test_transform = transforms.Compose([
    transforms.Resize(236, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# -- Cache --
@st.cache_resource
def load_ER_model():
    model = models.convnext_tiny(weights='DEFAULT')
    return model

@st.cache_resource
def load_PR_model():
    model = models.convnext_tiny(weights='DEFAULT')
    return model

# -- FRAGMENTS --
@st.fragment
def ER_details():
    training_details = st.checkbox("Show training details")
    if training_details:
        ER_performance = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Score'],
            'ConvNexT': [0.989, 0.994, 0.993, 0.993, 0.982],
            'ConvNexT w/ cell count': [0.986, 0.988, 0.994, 0.991, 0.969]
        })
        st.dataframe(ER_performance, hide_index=True, use_container_width=True)
        st.write("Training Phase: ")
        st.write("ConvNexT ")
        st.image('assets/ER_loss1.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/ER_cellcount_loss1.png', use_container_width=True)
        st.write("Fine-Tuning Phase: ")
        st.write("ConvNexT ")
        st.image('assets/ER_loss2.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/ER_cellcount_loss2.png', use_container_width=True)
        st.write("Confusion Matrix on Test Data: ")
        st.write("ConvNexT ")
        st.image('assets/ER_loss3.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/ER_cellcount_loss3.png', use_container_width=True)
        st.write("---")

@st.fragment
def ER_image_selection():
    st.subheader("2. Browse our image gallery or upload your own image:")
    ER_image_dir = 'assets/ER_images'
    # loop through the images in the directory and create a dictionary with title=file_name and img=image_path
    ER_image_array = []
    for file_name in os.listdir(ER_image_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_title = file_name.split('.')[0]
            img_dict = dict({
                "title": file_title,
                "img": os.path.join(ER_image_dir, file_name)
            })
            ER_image_array.append(img_dict)

    # sort ER_image_array by title
    ER_image_array = sorted(ER_image_array, key=lambda x: x["title"])
    st.session_state.ER_image_array = ER_image_array

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(ER_image_array[0]["img"], use_container_width=True)
        st.write(ER_image_array[0]["title"])
        st.image(ER_image_array[1]["img"], use_container_width=True)
        st.write(ER_image_array[1]["title"])
    with col2:
        st.image(ER_image_array[2]["img"], use_container_width=True)
        st.write(ER_image_array[2]["title"])
        st.image(ER_image_array[3]["img"], use_container_width=True)
        st.write(ER_image_array[3]["title"])
    with col3:
        st.image(ER_image_array[4]["img"], use_container_width=True)
        st.write(ER_image_array[4]["title"])
        st.image(ER_image_array[5]["img"], use_container_width=True)
        st.write(ER_image_array[5]["title"])
    with col4:
        st.image(ER_image_array[6]["img"], use_container_width=True)
        st.write(ER_image_array[6]["title"])
        st.image(ER_image_array[7]["img"], use_container_width=True)
        st.write(ER_image_array[7]["title"])
    with col5:
        st.image(ER_image_array[8]["img"], use_container_width=True)
        st.write(ER_image_array[8]["title"])
        st.image(ER_image_array[9]["img"], use_container_width=True)
        st.write(ER_image_array[9]["title"])

    col1, col2 = st.columns((1, 2))
    with col1:
        st.session_state.image_selection_gallery = st.selectbox("Select an image or...", [img["title"] for img in ER_image_array])
    with col2:
        st.session_state.image_selection_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.write("---")

@st.fragment
def ER_prediction():
    st.subheader("3. Run Analysis: ")
    if st.button("Run Analysis"):
        with st.spinner("Running Cell Counting Module..."):
            if st.session_state.image_selection_upload is not None:
                uploaded_image = st.session_state.image_selection_upload
            else:
                # load the selected image from the gallery
                for img in st.session_state.ER_image_array:
                    if st.session_state.image_selection_gallery == img["title"]:
                        uploaded_image = img["img"]
                        break
            cell_count, gray, blur, canny, dilated, rgb = canny_count(Image.open(uploaded_image), show_img=True)
            st.write("Original Image: ")
            st.image(uploaded_image, use_container_width=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(gray, caption="Gray Image", use_container_width=True)
            with col2:
                st.image(blur, caption="Blurred Image", use_container_width=True)
            with col3:
                st.image(canny, caption="Canny Edge Detection", use_container_width=True)
            with col4:
                st.image(dilated, caption="Dilated Image", use_container_width=True)
            with col5:
                st.image(rgb, caption="Contours", use_container_width=True)
            st.success(f"Cell Count: {cell_count}")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Running CNN Model..."):
                model = load_ER_model()
                model.classifier = nn.Sequential(
                    nn.LayerNorm([768,1,1], eps=1e-06, elementwise_affine=True),
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(in_features=768, out_features=2, bias=True)
                )
                model.load_state_dict(torch.load('assets/ER_weight.pth', weights_only=True, map_location=torch.device('cpu')))
                model.eval()
                st.success("CNN Model Successfully Loaded")
                # Transform original image using test_transform
                image = Image.open(uploaded_image).convert('RGB')
                image_tensor = test_transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")
        with col2:
            with st.spinner("Running Model with Cell Count..."):
                model_cellcount = load_ER_model()
                model_cellcount = ConvNextWithCellCount(base_model=model, cell_count_dim=1, num_classes=2)
                model_cellcount.load_state_dict(torch.load('assets/ER_weight_cellcount.pth', weights_only=True, map_location=torch.device('cpu')))
                model_cellcount.eval()
                st.success("CNN Model with Cell Count Successfully Loaded")
                # normalize cell count Cell count mean: 1258.9178554993098, count std: 1139.754365120176
                cell_count = (cell_count - 1258.9178554993098) / 1139.754365120176
                with torch.no_grad():
                    output = model_cellcount(image_tensor, torch.tensor([cell_count]))
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")


@st.fragment
def PR_details():
    training_details = st.checkbox("Show training details")
    if training_details:
        ER_performance = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Score'],
            'ConvNexT': [0.999, 1, 0.999, 0.999, 0.999],
            'ConvNexT w/ cell count': [1, 1, 1, 1, 1]
        })
        st.dataframe(ER_performance, hide_index=True, use_container_width=True)
        st.write("Training Phase: ")
        st.write("ConvNexT ")
        st.image('assets/PR_loss1.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/PR_cellcount_loss1.png', use_container_width=True)
        st.write("Fine-Tuning Phase: ")
        st.write("ConvNexT ")
        st.image('assets/PR_loss2.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/PR_cellcount_loss2.png', use_container_width=True)
        st.write("Confusion Matrix on Test Data: ")
        st.write("ConvNexT ")
        st.image('assets/PR_loss3.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/PR_cellcount_loss3.png', use_container_width=True)
        st.write("---")

@st.fragment
def PR_image_selection():
    st.subheader("2. Browse our image gallery or upload your own image:")
    PR_image_dir = 'assets/PR_images'
    # loop through the images in the directory and create a dictionary with title=file_name and img=image_path
    PR_image_array = []
    for file_name in os.listdir(PR_image_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_title = file_name.split('.')[0]
            img_dict = dict({
                "title": file_title,
                "img": os.path.join(PR_image_dir, file_name)
            })
            PR_image_array.append(img_dict)

    # sort ER_image_array by title
    PR_image_array = sorted(PR_image_array, key=lambda x: x["title"])
    st.session_state.PR_image_array = PR_image_array

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(PR_image_array[0]["img"], use_container_width=True)
        st.write(PR_image_array[0]["title"])
        st.image(PR_image_array[1]["img"], use_container_width=True)
        st.write(PR_image_array[1]["title"])
    with col2:
        st.image(PR_image_array[2]["img"], use_container_width=True)
        st.write(PR_image_array[2]["title"])
        st.image(PR_image_array[3]["img"], use_container_width=True)
        st.write(PR_image_array[3]["title"])
    with col3:
        st.image(PR_image_array[4]["img"], use_container_width=True)
        st.write(PR_image_array[4]["title"])
        st.image(PR_image_array[5]["img"], use_container_width=True)
        st.write(PR_image_array[5]["title"])
    with col4:
        st.image(PR_image_array[6]["img"], use_container_width=True)
        st.write(PR_image_array[6]["title"])
        st.image(PR_image_array[7]["img"], use_container_width=True)
        st.write(PR_image_array[7]["title"])
    with col5:
        st.image(PR_image_array[8]["img"], use_container_width=True)
        st.write(PR_image_array[8]["title"])
        st.image(PR_image_array[9]["img"], use_container_width=True)
        st.write(PR_image_array[9]["title"])

    col1, col2 = st.columns((1, 2))
    with col1:
        st.session_state.image_selection_gallery = st.selectbox("Select an image or...", [img["title"] for img in PR_image_array])
    with col2:
        st.session_state.image_selection_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.write("---")

@st.fragment
def PR_prediction():
    st.subheader("3. Run Analysis: ")
    if st.button("Run Analysis"):
        with st.spinner("Running Cell Counting Module..."):
            if st.session_state.image_selection_upload is not None:
                uploaded_image = st.session_state.image_selection_upload
            else:
                # load the selected image from the gallery
                for img in st.session_state.PR_image_array:
                    if st.session_state.image_selection_gallery == img["title"]:
                        uploaded_image = img["img"]
                        break
            cell_count, gray, blur, canny, dilated, rgb = canny_count(Image.open(uploaded_image), show_img=True)
            st.write("Original Image: ")
            st.image(uploaded_image, use_container_width=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(gray, caption="Gray Image", use_container_width=True)
            with col2:
                st.image(blur, caption="Blurred Image", use_container_width=True)
            with col3:
                st.image(canny, caption="Canny Edge Detection", use_container_width=True)
            with col4:
                st.image(dilated, caption="Dilated Image", use_container_width=True)
            with col5:
                st.image(rgb, caption="Contours", use_container_width=True)
            st.success(f"Cell Count: {cell_count}")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Running CNN Model..."):
                model = load_PR_model()
                model.classifier = nn.Sequential(
                    nn.LayerNorm([768,1,1], eps=1e-06, elementwise_affine=True),
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(in_features=768, out_features=2, bias=True)
                )
                model.load_state_dict(torch.load('assets/PR_weight.pth', weights_only=True, map_location=torch.device('cpu')))
                model.eval()
                st.success("CNN Model Successfully Loaded")
                # Transform original image using test_transform
                image = Image.open(uploaded_image).convert('RGB')
                image_tensor = test_transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")
        with col2:
            with st.spinner("Running Model with Cell Count..."):
                model_cellcount = load_PR_model()
                model_cellcount = ConvNextWithCellCount(base_model=model, cell_count_dim=1, num_classes=2)
                model_cellcount.load_state_dict(torch.load('assets/PR_weight_cellcount.pth', weights_only=True, map_location=torch.device('cpu')))
                model_cellcount.eval()
                st.success("CNN Model with Cell Count Successfully Loaded")
                # normalize cell count Cell count mean: 2211.7708333333335, count std: 1940.2254832315882
                cell_count = (cell_count - 2211.7708333333335) / 1940.2254832315882
                with torch.no_grad():
                    output = model_cellcount(image_tensor, torch.tensor([cell_count]))
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")


@st.fragment
def Ki_details():
    training_details = st.checkbox("Show training details")
    if training_details:
        ER_performance = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Score'],
            'ConvNexT': [0.979, 0.903, 0.8, 0.848, 0.896],
            'ConvNexT w/ cell count': [0.987, 0.885, 0.939, 0.911, 0.965]
        })
        st.dataframe(ER_performance, hide_index=True, use_container_width=True)
        st.write("Training Phase: ")
        st.write("ConvNexT ")
        st.image('assets/Ki_loss1.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/Ki_cellcount_loss1.png', use_container_width=True)
        st.write("Fine-Tuning Phase: ")
        st.write("ConvNexT ")
        st.image('assets/Ki_loss2.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/Ki_cellcount_loss2.png', use_container_width=True)
        st.write("Confusion Matrix on Test Data: ")
        st.write("ConvNexT ")
        st.image('assets/Ki_loss3.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/Ki_cellcount_loss3.png', use_container_width=True)
        st.write("---")

@st.fragment
def Ki_image_selection():
    st.subheader("2. Browse our image gallery or upload your own image:")
    Ki_image_dir = 'assets/Ki_images'
    # loop through the images in the directory and create a dictionary with title=file_name and img=image_path
    Ki_image_array = []
    for file_name in os.listdir(Ki_image_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_title = file_name.split('.')[0]
            img_dict = dict({
                "title": file_title,
                "img": os.path.join(Ki_image_dir, file_name)
            })
            Ki_image_array.append(img_dict)

    # sort ER_image_array by title
    Ki_image_array = sorted(Ki_image_array, key=lambda x: x["title"])
    st.session_state.Ki_image_array = Ki_image_array

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(Ki_image_array[0]["img"], use_container_width=True)
        st.write(Ki_image_array[0]["title"])
        st.image(Ki_image_array[1]["img"], use_container_width=True)
        st.write(Ki_image_array[1]["title"])
    with col2:
        st.image(Ki_image_array[2]["img"], use_container_width=True)
        st.write(Ki_image_array[2]["title"])
        st.image(Ki_image_array[3]["img"], use_container_width=True)
        st.write(Ki_image_array[3]["title"])
    with col3:
        st.image(Ki_image_array[4]["img"], use_container_width=True)
        st.write(Ki_image_array[4]["title"])
        st.image(Ki_image_array[5]["img"], use_container_width=True)
        st.write(Ki_image_array[5]["title"])
    with col4:
        st.image(Ki_image_array[6]["img"], use_container_width=True)
        st.write(Ki_image_array[6]["title"])
        st.image(Ki_image_array[7]["img"], use_container_width=True)
        st.write(Ki_image_array[7]["title"])
    with col5:
        st.image(Ki_image_array[8]["img"], use_container_width=True)
        st.write(Ki_image_array[8]["title"])
        st.image(Ki_image_array[9]["img"], use_container_width=True)
        st.write(Ki_image_array[9]["title"])

    col1, col2 = st.columns((1, 2))
    with col1:
        st.session_state.image_selection_gallery = st.selectbox("Select an image or...", [img["title"] for img in Ki_image_array])
    with col2:
        st.session_state.image_selection_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.write("---")

@st.fragment
def Ki_prediction():
    st.subheader("3. Run Analysis: ")
    if st.button("Run Analysis"):
        with st.spinner("Running Cell Counting Module..."):
            if st.session_state.image_selection_upload is not None:
                uploaded_image = st.session_state.image_selection_upload
            else:
                # load the selected image from the gallery
                for img in st.session_state.Ki_image_array:
                    if st.session_state.image_selection_gallery == img["title"]:
                        uploaded_image = img["img"]
                        break
            cell_count, gray, blur, canny, dilated, rgb = canny_count(Image.open(uploaded_image), show_img=True)
            st.write("Original Image: ")
            st.image(uploaded_image, use_container_width=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.image(gray, caption="Gray Image", use_container_width=True)
            with col2:
                st.image(blur, caption="Blurred Image", use_container_width=True)
            with col3:
                st.image(canny, caption="Canny Edge Detection", use_container_width=True)
            with col4:
                st.image(dilated, caption="Dilated Image", use_container_width=True)
            with col5:
                st.image(rgb, caption="Contours", use_container_width=True)
            st.success(f"Cell Count: {cell_count}")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Running CNN Model..."):
                model = load_PR_model()
                model.classifier = nn.Sequential(
                    nn.LayerNorm([768,1,1], eps=1e-06, elementwise_affine=True),
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(in_features=768, out_features=2, bias=True)
                )
                model.load_state_dict(torch.load('assets/Ki67_weight.pth', weights_only=True, map_location=torch.device('cpu')))
                model.eval()
                st.success("CNN Model Successfully Loaded")
                # Transform original image using test_transform
                image = Image.open(uploaded_image).convert('RGB')
                image_tensor = test_transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'LOW' if prediction == 1 else 'HIGH'}")
                st.write(f"Probability: {prob}")
        with col2:
            with st.spinner("Running Model with Cell Count..."):
                model_cellcount = load_PR_model()
                model_cellcount = ConvNextWithCellCount(base_model=model, cell_count_dim=1, num_classes=2)
                model_cellcount.load_state_dict(torch.load('assets/Ki67_weight_cellcount.pth', weights_only=True, map_location=torch.device('cpu')))
                model_cellcount.eval()
                st.success("CNN Model with Cell Count Successfully Loaded")
                # normalize cell count Cell count mean: 2211.7708333333335, count std: 1940.2254832315882
                cell_count = (cell_count - 2211.7708333333335) / 1940.2254832315882
                with torch.no_grad():
                    output = model_cellcount(image_tensor, torch.tensor([cell_count]))
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'LOW' if prediction == 1 else 'HIGH'}")
                st.write(f"Probability: {prob}")


@st.fragment
def HER_details():
    training_details = st.checkbox("Show training details")
    if training_details:
        ER_performance = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC Score'],
            'ConvNexT': [1, 1, 1, 1, 1],
            'ConvNexT w/ cell count': [1, 1, 1, 1, 1]
        })
        st.dataframe(ER_performance, hide_index=True, use_container_width=True)
        st.write("Training Phase: ")
        st.write("ConvNexT ")
        st.image('assets/HER_loss1.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/HER_cellcount_loss1.png', use_container_width=True)
        st.write("Fine-Tuning Phase: ")
        st.write("ConvNexT ")
        st.image('assets/HER_loss2.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/HER_cellcount_loss2.png', use_container_width=True)
        st.write("Confusion Matrix on Test Data: ")
        st.write("ConvNexT ")
        st.image('assets/HER_loss3.png', use_container_width=True)
        st.write("ConvNexT w/ cell count ")
        st.image('assets/HER_cellcount_loss3.png', use_container_width=True)
        st.write("---")

@st.fragment
def HER_image_selection():
    st.subheader("2. Browse our image gallery or upload your own image:")
    HER_image_dir = 'assets/HER_images'
    # loop through the images in the directory and create a dictionary with title=file_name and img=image_path
    HER_image_array = []
    for file_name in os.listdir(HER_image_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            file_title = file_name.split('.')[0]
            img_dict = dict({
                "title": file_title,
                "img": os.path.join(HER_image_dir, file_name)
            })
            HER_image_array.append(img_dict)

    # sort ER_image_array by title
    HER_image_array = sorted(HER_image_array, key=lambda x: x["title"])
    st.session_state.HER_image_array = HER_image_array

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(HER_image_array[0]["img"], use_container_width=True)
        st.write(HER_image_array[0]["title"])
    with col2:
        st.image(HER_image_array[1]["img"], use_container_width=True)
        st.write(HER_image_array[1]["title"])
    with col3:
        st.image(HER_image_array[2]["img"], use_container_width=True)
        st.write(HER_image_array[2]["title"])
    with col4:
        st.image(HER_image_array[3]["img"], use_container_width=True)
        st.write(HER_image_array[3]["title"])
    with col5:
        st.image(HER_image_array[4]["img"], use_container_width=True)
        st.write(HER_image_array[4]["title"])

    col1, col2 = st.columns((1, 2))
    with col1:
        st.session_state.image_selection_gallery = st.selectbox("Select an image or...", [img["title"] for img in HER_image_array])
    with col2:
        st.session_state.image_selection_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.write("---")

@st.fragment
def HER_prediction():
    st.subheader("3. Run Analysis: ")
    if st.button("Run Analysis"):
        with st.spinner("Running Cell Counting Module..."):
            if st.session_state.image_selection_upload is not None:
                uploaded_image = st.session_state.image_selection_upload
            else:
                # load the selected image from the gallery
                for img in st.session_state.HER_image_array:
                    if st.session_state.image_selection_gallery == img["title"]:
                        uploaded_image = img["img"]
                        break
            cell_area, gray, binary_mask = process_image(Image.open(uploaded_image), show_img=True) 
            st.write("Original Image: ")
            st.image(uploaded_image, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(gray, caption="Gray Image", use_container_width=True)
            with col2:
                st.image(binary_mask, caption="Binary Mask", use_container_width=True)

            st.success(f"Cell area: {cell_area}")
        col1, col2 = st.columns(2)
        with col1:
            with st.spinner("Running CNN Model..."):
                model = load_PR_model()
                model.classifier = nn.Sequential(
                    nn.LayerNorm([768,1,1], eps=1e-06, elementwise_affine=True),
                    nn.Flatten(start_dim=1, end_dim=-1),
                    nn.Linear(in_features=768, out_features=2, bias=True)
                )
                model.load_state_dict(torch.load('assets/HER_weight.pth', weights_only=True, map_location=torch.device('cpu')))
                model.eval()
                st.success("CNN Model Successfully Loaded")
                # Transform original image using test_transform
                image = Image.open(uploaded_image).convert('RGB')
                image_tensor = test_transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(image_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")
        with col2:
            with st.spinner("Running Model with Cell Count..."):
                model_cellcount = load_PR_model()
                model_cellcount = ConvNextWithCellCount(base_model=model, cell_count_dim=1, num_classes=2)
                model_cellcount.load_state_dict(torch.load('assets/HER_weight_cellcount.pth', weights_only=True, map_location=torch.device('cpu')))
                model_cellcount.eval()
                st.success("CNN Model with Cell Count Successfully Loaded")
                # normalize cell count Cell count mean: 2211.7708333333335, count std: 1940.2254832315882
                cell_area = (cell_area - 0.3283004405082788) / 0.17266037240662982
                with torch.no_grad():
                    output = model_cellcount(image_tensor, torch.tensor([cell_area]))
                    prediction = torch.argmax(output, dim=1).item()
                    prob = torch.nn.functional.softmax(output, dim=1)[0].tolist()
                st.write(f"Prediction: {'Positive' if prediction == 1 else 'Negative'}")
                st.write(f"Probability: {prob}")