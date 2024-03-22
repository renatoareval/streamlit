from PIL import Image
import torch
import os
import streamlit as st
from datetime import datetime

def imageInput(device, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Salvar a imagem temporariamente
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', f"{ts}_{image_file.name}")
            img.save(imgpath)

            # Carregar o modelo YOLOv5
            model = torch.hub.load("ultralytics/yolov5", "yolov5s")
            model.to(device)

            
            results = model(imgpath)

           
            st.image(results.render(), caption='Model Prediction(s)', use_column_width=True)

def main():
   
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data.'])
    
    option = st.sidebar.radio("Select input type.", ['Image'], index=0)
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu'], index=0)

   
    st.header('📦Obstacle Detection Model Demo')
    st.subheader('👈🏽 Select the options')
    st.sidebar.markdown("https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")
    
    if option == "Image":    
        imageInput(deviceoption, datasrc)

if __name__ == '__main__':
    main()
