import cv2
import numpy as np
from PIL import Image
import streamlit as st

from ml.ml_projects.mnist_classification.incremental_inference import run_for_uploaded_img

if __name__=='__main__':
    st.title("Digit Classifier")
    st.write("Upload an image of a handwritten digit, and the app will classify it.")

    uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        pil_img = Image.open(uploaded_file).convert('L')
        print(pil_img.size)

        st.image(pil_img,width=400, caption="Uploaded Image", use_column_width=False)
        confidence, pred_label = run_for_uploaded_img(pil_img=pil_img)
        st.write(f"Predicted Digit: {pred_label.item()}")
        st.write(f"Confidence: {int(100*confidence.item())}%")

        st.write("If the prediction is incorrect, please provide the correct input")
        st.number_input('Correct Label (0-9)',max_value=9,min_value=0,step=1,format="%d")
        if st.button("Use for Retraining"):
            pass