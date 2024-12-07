import cv2
import numpy as np
from PIL import Image
import streamlit as st

from single_inference import run as run_for_uploaded_img
from retraining import run as retrain_for_single_image

if __name__=='__main__':
    st.title("Digit Classifier")
    st.write("Upload an image of a handwritten digit, and the app will classify it.")

    uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(image,width=400, caption="Uploaded Image", use_column_width=False)
        confidence, pred_label = run_for_uploaded_img(img=image)
        st.write(f"Predicted Digit: {pred_label.item()}")
        st.write(f"Confidence: {int(100*confidence.item())}%")

        st.title("Retraining")
        st.write("If the above prediction is incorrect, please provide the correct input")
        correct_label = st.number_input('Correct Label (0-9)',max_value=9,min_value=0,step=1,format="%d")
        print(correct_label,pred_label)
        if st.button("Use for Retraining"):
            if correct_label != pred_label:
                retrain_for_single_image(img=image,label=correct_label)