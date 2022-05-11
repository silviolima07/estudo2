import base64
import json
import os
import re
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
#from svgpathtools import parse_path

from tensorflow import keras

modelo_keras = keras.models.load_model('./modelo_keras.h5')

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        #"About": about,
        "Basic example": full_app,
        #"Get center coords of circles": center_circle_app,
        #"Color-based image annotation": color_annotation_app,
        #"Download Base64 encoded PNG": png_export,
        #"Compute the length of drawn arcs": compute_arc_length,
        "Draw numbers from 0 to 9": png_export,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

   





def full_app():
    st.sidebar.header("Configuration")

    with st.echo("below"):
        # Specify canvas parameters in application
        drawing_mode = st.sidebar.selectbox(
           "Drawing tool:",
            ("freedraw"),
        )
        stroke_width = 10 #st.sidebar.slider("Stroke width: ", 1, 25, 3)
            
        stroke_color = "rgba(255, 255, 255)"
        bg_color = "rgba(0, 0, 0)"
        bg_image =  st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
        realtime_update = st.sidebar.checkbox("Update in realtime", True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=600,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            display_toolbar=st.sidebar.checkbox("Display toolbar", True),
            key="full_app",
        )

        # Do something interesting with the image data and paths
        
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
        #if canvas_result.json_data is not None:
        #    objects = pd.json_normalize(canvas_result.json_data["objects"])
        #    for col in objects.select_dtypes(include=["object"]).columns:
        #        objects[col] = objects[col].astype("str")
        #    st.dataframe(objects)
        
            img_data = canvas_result.image_data
            im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")      
            img_28_28 = im.resize([28,28], Image.Resampling.NEAREST)
            st.image(img_28_28)
            img_array = np.array(img_28_28)
            img_784 = img_array.reshape(-1,28*28)
            img_784 = img_784.astype('float32')
            img_normalizado = img_784/255.0
            
            if st.button("Prever")and canvas_result.image_data is not None:
             
                st.title("Previsão")
                 
                pred = modelo_keras.predict(img_normalizado)
                st.write(pred)
                st.title(pred.argmax())



def png_export():
    

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        button_id = st.session_state["button_id"]
        file_path = f"tmp/{button_id}.png"
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        
        if st.button("Prever")and data is not None:
             
                st.title("Previsão")
                 
                pred = modelo_keras.predict(img_normalizado)
                st.write(pred)
                st.title(pred.argmax())




if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    
    
    #mnist_keras = keras.models.load_model('./mnist_keras.h5')
    
    main()
