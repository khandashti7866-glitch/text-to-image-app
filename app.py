import streamlit as st
from PIL import Image
import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline

# Page config
st.set_page_config(page_title="Text-to-Image AI", layout="wide")
st.title("🎨 AI Text-to-Image Generator (CPU/GPU Compatible)")

# Sidebar settings
st.sidebar.header("Settings")
width = st.sidebar.selectbox("Image Width", [256, 512, 768])
height = st.sidebar.selectbox("Image Height", [256, 512, 768])
num_images = st.sidebar.slider("Number of Images", 1, 3, 1)
style = st.sidebar.selectbox("Art Style", ["Realistic", "Cartoon", "Anime", "Digital Art"])

# Prompt input
prompt = st.text_area("Enter your prompt", "")

# Generate button
generate_button = st.button("Generate Image")

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device=="cuda" else torch.float32
    )
    pipe = pipe.to(device)
    return pipe

if generate_button:
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating images..."):
            try:
                pipe = load_model()
                images = []
                for _ in range(num_images):
                    img = pipe(f"{prompt}, {style}", width=width, height=height, guidance_scale=7.5).images[0]
                    images.append(img)
                
                cols = st.columns(len(images))
                for i, img in enumerate(images):
                    cols[i].image(img, use_column_width=True)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    cols[i].download_button("Download", buf.getvalue(), file_name=f"image_{i+1}.png")
            except Exception as e:
                st.error(f"Error: {e}")
