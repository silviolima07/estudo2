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
from svgpathtools import parse_path

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
        #    st.image(canvas_result.image_data)
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
    st.markdown(
        """
    Realtime update is disabled for this demo. 
    Press the 'Download' button at the bottom of canvas to update exported image.
    """
    )
    try:
        Path("tmp/").mkdir()
    except FileExistsError:
        pass

    # Regular deletion of tmp files
    # Hopefully callback makes this better
    now = time.time()
    N_HOURS_BEFORE_DELETION = 1
    for f in Path("tmp/").glob("*.png"):
        st.write(f, os.stat(f).st_mtime, now)
        if os.stat(f).st_mtime < now - N_HOURS_BEFORE_DELETION * 3600:
            Path.unlink(f)

    if st.session_state["button_id"] == "":
        st.session_state["button_id"] = re.sub(
            "\d+", "", str(uuid.uuid4()).replace("-", "")
        )

    button_id = st.session_state["button_id"]
    file_path = f"tmp/{button_id}.png"

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(0,0,0);
                color: rgb(255,255,255);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    data = st_canvas(update_streamlit=False, key="png_export")
    if data is not None and data.image_data is not None:
        img_data = data.image_data
        im = Image.fromarray(img_data.astype("uint8"), mode="RGBA")
        im.save(file_path, "PNG")

        buffered = BytesIO()
        im.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(img_data.encode()).decode()
        except AttributeError:
            b64 = base64.b64encode(img_data).decode()

        dl_link = (
            custom_css
            + f'<a download="{file_path}" id="{button_id}" href="data:file/txt;base64,{b64}">Export PNG</a><br></br>'
        )
        st.markdown(dl_link, unsafe_allow_html=True)




if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Drawable Canvas Demo")
    st.sidebar.subheader("Configuration")
    
    
    #mnist_keras = keras.models.load_model('./mnist_keras.h5')
    
    main()
