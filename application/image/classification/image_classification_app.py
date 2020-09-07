import streamlit as st
from ...utils import DatasetDownloader


def image_classification_app():
    action_option = st.selectbox(
        'How would you want to give us the dataset?',
        ('', 'URL', 'Upload Zip', 'Kaggle Dataset')
    )
    if action_option == 'URL':
        url = st.text_input('Enter URL', '')
        if url != '':
            st.text('Hold tight, we are getting you dataset...')
            dataset_path = DatasetDownloader.download_from_url(url)
            st.text('Done!')
            print(dataset_path)
    elif action_option == 'Upload Zip':
        st.warning('Under Development')
    elif action_option == 'Kaggle Dataset':
        st.warning('Under Development')
    elif action_option != '':
        st.error('Action Option Not recognized')