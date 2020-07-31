from PIL import Image
import streamlit as st
import tensorflow as tf
from .models import MODEL_DICT
from src.gradcam import GradCam
from .utils import (
    get_overlayed_image, get_model_summary,
    visualize_classification_prediction
)


def model_explorer(model):
    st.markdown(
        '<hr><h3>Model Explorer</h3><br>',
        unsafe_allow_html=True
    )
    model_json = model.to_json()
    model_explorer_checkbox = st.checkbox(
        'Maximize', value=True, key='model_explorer_checkbox'
    )
    if model_explorer_checkbox:
        st.json(model_json)


def visualize_prediction_probablities(prediction):
    st.markdown(
        '<hr><h3>Prediction Probablities</h3><br>',
        unsafe_allow_html=True
    )
    plotly_figure = visualize_classification_prediction(prediction)
    st.plotly_chart(plotly_figure, use_container_width=True)


def visualize_gradcam(
        image_pil, model, preprocess_function, decode_function,
        last_layer, classifier_layers, size, original_size):

    grad_cam = GradCam(
        model, preprocess_function, decode_function, last_layer=last_layer,
        classifier_layers=classifier_layers, size=size
    )
    prediction = grad_cam.get_prediction(image_pil)
    visualize_prediction_probablities(prediction)

    st.markdown(
        '<hr><h3>GradCam</h3><br>',
        unsafe_allow_html=True
    )

    image_tensor = grad_cam.get_tensor(image_pil)
    gradcam_heatmap = grad_cam.apply_gradcam(image_tensor)

    overlayed_image = get_overlayed_image(
        image_pil, gradcam_heatmap,
        original_size, weightage=0.7
    )

    st.image(overlayed_image)


def inference_branch_app():
    # ./assets/cat_1.jpg
    image_path = st.sidebar.file_uploader('Please Select a File')
    if image_path is not None:
        try:
            image_pil = Image.open(image_path)
            original_size = image_pil.size
            height, width = original_size
            image_pil = image_pil.resize((224, 224))
            st.image(image_pil, caption='Input Image')
        except:
            st.error('Invalid File')

        model_option = st.sidebar.selectbox(
            'Please Select the Model',
            tuple([''] + list(MODEL_DICT.keys()))
        )

        if model_option in list(MODEL_DICT.keys()):

            (
                model, preprocess_function, decode_function,
                last_layer, size, classifier_layers
            ) = MODEL_DICT[model_option]
            model = model(weights='imagenet')
            model_explorer(model)

            classify_button = st.sidebar.button('Classify')

            if classify_button:
                visualize_gradcam(
                    image_pil, model, preprocess_function, decode_function,
                    last_layer, classifier_layers, size, original_size
                )
