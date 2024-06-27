import streamlit as st
from PIL import Image

def main():
    st.title("Image Uploader and Viewer")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a JPEG image file", type="jpg")
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
    
if __name__ == '__main__':
    main()