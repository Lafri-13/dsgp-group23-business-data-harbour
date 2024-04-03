import streamlit as st
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video
import torch
import tempfile
import os

# ======================================= Content Generation Method Definitions =======================================

def text_to_image(prompt, negative_prompt, num_inference_steps=50):
    # Create an instance of the StableDiffusion pipeline (one-time for efficiency)
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Define generation parameters
    generation_inputs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
    }

    # Generate the image
    images = pipe(**generation_inputs)["images"]

    # Return the first generated image (can be adjusted based on needs)
    return images[0]

# =====================================================================================================================

def image_to_image(prompt, uploaded_image=None, strength=0.75, guidance_scale=2.5, num_inference_steps=50):
    # Define the pipeline and scheduler
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # Use uploaded image if provided
    if uploaded_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Check if uploaded_image has content before reading
            if uploaded_image.read():
                temp_file.write(uploaded_image.read())
                temp_file.seek(0)  # Reset file pointer for image reading by pipeline
            else:
                raise ValueError("Please upload a valid image file.")
            image_path = temp_file.name  # Store the temporary image path

        # Generate the image with prompt and image path
        with torch.autocast("cuda"):
            images = pipe(
                prompt=prompt,
                init_image=image_path,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                scheduler=scheduler,
            )

        # Clean up the temporary file after processing
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
    if content_option == "Text to Image":
        with st.form("text_to_image"):
            prompt = st.text_input("Please enter your prompt :")
            negative_prompt = st.text_input("Please enter your negative prompt :")
            submit_text_to_image = st.form_submit_button("Generate Image")
            if submit_text_to_image:
                st.image(text_to_image(prompt, negative_prompt))

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

# Display the selected form
show_form(selected_form)


