import streamlit as st
import numpy as np
import pandas as pd
import os
from fragments.main_fragments import ER_details, ER_image_selection, ER_prediction, PR_details, PR_image_selection, PR_prediction, Ki_details, Ki_image_selection, Ki_prediction, HER_details, HER_image_selection, HER_prediction


# -- PAGE CONFIG --
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="⚕️"
)

# -- Session State --
if 'image_selection_upload' not in st.session_state:
    st.session_state.image_selection_upload = None

if 'image_selection_gallery' not in st.session_state:
    st.session_state.image_selection_gallery = None

if 'ER_image_array' not in st.session_state:
    st.session_state.ER_image_array = None

if 'PR_image_array' not in st.session_state:
    st.session_state.PR_image_array = None

if 'Ki_image_array' not in st.session_state:
    st.session_state.Ki_image_array = None

# -- PAGE SETUP --
st.title("⚕️ Breast Cancer Detection")
st.subheader("⚙️ Model Architecture")
st.markdown('<div style="text-align: justify;">The CNN model used is ConvNext-Tiny. Images will be resized to 224x224 to fit into the CNN, but the original images will also be passed to the \"Cell Counting\" module. This module, for ER, PR, and Ki67, uses Canny edge detection and produces a count (e.g., 1200, 2810, 300, etc.). For HER2, it calculates the percentage of the brown area (e.g., 0.8, 0.756, 0.201). The numbers obtained from this module are normalized before being combined with 768 extracted features from the CNN. These 769 features are then fed into a Fully Connected layer, ultimately producing the CNN output.</div>', unsafe_allow_html=True)
st.image('assets/hero_image.png', use_container_width=True)
st.write("---")

st.subheader("1. Choose a model to use:")
task_selection = st.radio("Select a task", ["ER", "PR", "Ki67", "HER2"], horizontal=True)

if task_selection == "ER":
    ER_details()
    ER_image_selection()
    ER_prediction()
elif task_selection == "PR":
    PR_details()
    PR_image_selection()
    PR_prediction()
elif task_selection == "Ki67":
    Ki_details()
    Ki_image_selection()
    Ki_prediction()
else:
    HER_details()
    HER_image_selection()
    HER_prediction()


