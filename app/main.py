import streamlit as st
import cv2
import numpy as np
from backend import predict, load_model, make_prediction,parse_equations


@st.cache_resource(ttl=3600)
def load_model():
    model = load_model()
    return model
    
image = None

st.title("Image Calculator")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png","webp","heic"])
if uploaded_file is not None:
    # Process the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, channels="RGB")
    equation = make_prediction(image)
    equation = parse_equations(equation)
    print(f"{equation=}")
    try:
        solve = eval(equation)
    except Exception as e:
        solve = f"Invalid Equation {e}"
    st.text(f"Prediction : {equation} = {solve}")
    # print(prediction)
    # st.image(prediction[0], channels="RGB")
    # Perform image processing operations on the image

if st.button("Capture Image"):
    # Capture image from webcam
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if ret:
        # Perform image processing operations on the captured image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Display the captured image
        st.image(image, channels="RGB")
    cap.release()

