import streamlit as st
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video
import torch
import tempfile
import os
from PIL import Image
import io

# ======================================= Content Generation Method Definitions =======================================

def text_to_image(prompt, negative_prompt, num_inference_steps=1):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    generation_inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }
    images = pipe(**generation_inputs)["images"]
    return images[0]

# =====================================================================================================================

def image_to_image(prompt, uploaded_image=None, strength=0.75, guidance_scale=2.5, num_inference_steps=50):
    # Define the pipeline and scheduler
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            if uploaded_image.read():
                temp_file.write(uploaded_image.read())
                temp_file.seek(0)  
            else:
                raise ValueError("Please upload a valid image file.")
            image_path = temp_file.name

        with torch.autocast("cuda"):
            images = pipe(
                prompt=prompt,
                init_image=image_path,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                scheduler=scheduler,
            )

        os.unlink(image_path)

        return images[0]
    else:
        raise ValueError("Please upload an image for processing.")

# =====================================================================================================================
    
def text_to_video(prompt, negative_prompt):
    pipe = DiffusionPipeline.from_pretrained("ali-vilab/text-to-video-ms-1.7b", torch_dtype = torch.float16, variant ='fp16')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

def show_form(content_option):
    # ======================================= Text to Image =======================================
    if content_option == "Text to Image" and dimension_option != "-- Select an option --":
        with st.form("text_to_image"):
            prompt = st.text_input("Please enter your prompt :")
            negative_prompt = st.text_input("Please enter your negative prompt :")
            image_width, image_height = get_dimensions(dimension_option)
            submit_text_to_image = st.form_submit_button("Generate Image")
            if submit_text_to_image:
                if image_width > 0 and image_height > 0:
                    st.image(image_resize_and_converter(text_to_image(prompt, negative_prompt), image_width, image_height))
                else:
                    st.write("Please enter a valid height and width for the image")

                

    # ======================================= Image to Image =======================================
    elif content_option == "Image to Image":
        with st.form("iamge_to_image"):
            prompt = st.text_input("Please enter your prompt :")
            negative_prompt = st.text_input("Please enter your negative prompt :")
            uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png"])
            submit_image_to_image = st.form_submit_button("Optimise Image")
            if submit_image_to_image:
                if uploaded_image is not None:
                    generated_image = image_to_image(prompt, uploaded_image=uploaded_image)
                    st.image(generated_image)  # Display the generated image
                else:
                    st.warning("Please upload an image to proceed.")

    # ======================================= Text to Video =======================================
    elif content_option == "Text to Video":
        with st.form("text_to_image"):
            prompt = st.text_input("Please enter your prompt :")
            negative_prompt = st.text_input("Please enter your negative prompt :")
            submit_text_to_video = st.form_submit_button("Generate Video")
            if submit_text_to_video:
                st.image(text_to_video(prompt, negative_prompt))

    # ======================================= Image to Video =======================================
    elif content_option == "Image to Video":
        with st.form("image_to_video"):
            message = st.text_area("Enter a message")
            submitted4 = st.form_submit_button("Submit")
            if submitted4:
                st.write(f"Your message: {message}")

    # ======================================= No Option Selected =======================================
    else:
        st.write("Please select the type of content you'd like to generate.")

# Create the dropdown menu
content_options = ["-- Select an option --", "Text to Image", "Image to Image", "Text to Video", "Image to Video"]
selected_form = st.selectbox("What kind of content would you like to generate today ?", content_options)

content_dimensions_options = ["-- Select an option --", "Instagram Story", "Instagram Post", "Facebook Post", "Youtube Video", "YouTube Shorts", "Custom Dimensions"]
dimension_option = st.selectbox("Please select the dimensions of the content you wish to create", content_dimensions_options)

# Creating the method to get and handle the user's input for the different dimensions of content they wish to create
def get_dimensions(dimension_option):
    if dimension_option == "Instagram Story":
        return 1080, 1920
    elif dimension_option == "Instagram Post":
        return 1080, 1080
    elif dimension_option == "Facebook Post":
        return 1080, 1080
    elif dimension_option == "Youtube Video":
        return 1280, 720
    elif dimension_option == "YouTube Shorts":
        return 1920, 1080
    elif dimension_option == "Custom Dimensions":
        content_height = st.number_input("Enter the height of the content : ")
        content_width = st.number_input("Enter the width of the content : ")
        return int(content_height), int(content_width)
    else:
        st.write("Please select the dimensions of the content you'd like to generate.")
        return 0,0

# Handling the image resizing and conversion from numpy to JPG using the PIL library
def image_resize_and_converter(image, width, height):
    resized_image = image.resize((width, height))
    return resized_image

show_form(selected_form)


