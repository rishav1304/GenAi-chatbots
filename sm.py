import streamlit as st
from googletrans import Translator
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
import imageio
from base64 import b64encode
import os

# Initialize the translator
translator = Translator()

# Configuration
class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (900, 900)
    image_gen_guidance_scale = 9
    video_gen_model_id = "damo-vilab/text-to-video-ms-1.7b"

# Load the models
@st.cache_resource
def load_image_gen_model():
    return StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id,
        torch_dtype=torch.float16,
        revision="fp16",
        use_auth_token=os.environ.get('YOUR_HUGGINGFACE_TOKEN')
    ).to(CFG.device)

@st.cache_resource
def load_video_gen_model():
    pipe = DiffusionPipeline.from_pretrained(CFG.video_gen_model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe

@st.cache_resource
def load_captioning_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

# Load all models
image_gen_model = load_image_gen_model()
video_gen_model = load_video_gen_model()
captioning_model, feature_extractor, tokenizer = load_captioning_model()

# Text translation
def translate_text(text, dest_lang="en"):
    translated_text = translator.translate(text, dest=dest_lang).text
    return translated_text

# Image generation function
def generate_image(prompt, model):
    image = model(prompt, num_inference_steps=CFG.image_gen_steps, generator=CFG.generator, guidance_scale=CFG.image_gen_guidance_scale).images[0]
    return image.resize(CFG.image_gen_size)

# Video generation function
def generate_video(prompt, model, duration_seconds=4):
    num_frames = duration_seconds * 10
    video_frames = model(prompt, negative_prompt="low quality", num_inference_steps=50, num_frames=num_frames).frames
    return video_frames

# Display video
def display_video(video_frames, fps=10):
    video_frames = video_frames.squeeze(0)
    modified_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    with imageio.get_writer('temp.mp4', fps=fps) as writer:
        for frame in modified_frames:
            writer.append_data(frame)
    with open('temp.mp4', 'rb') as f:
        video_data = f.read()
    video_url = f"data:video/mp4;base64,{b64encode(video_data).decode()}"
    st.video(video_url)

# Caption prediction
def predict_caption(image_paths, model, feature_extractor, tokenizer):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values.to(CFG.device)
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [caption.strip() for caption in captions]

# Streamlit Interface
st.title("ReImagination")

# Task selection
task = st.radio("Select a task", ["Text to Image", "Text to Video", "Image to Text"])

# Text to Image
if task == "Text to Image":
    prompt = st.text_input("Enter text for image generation:")
    if st.button("Generate Image"):
        if prompt:
            translated_prompt = translate_text(prompt, "en")
            st.write(f"Translated prompt: {translated_prompt}")
            with st.spinner("Generating image..."):
                generated_image = generate_image(translated_prompt, image_gen_model)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Please enter a prompt.")

# Text to Video
elif task == "Text to Video":
    prompt = st.text_input("Enter text for video generation:")
    duration = st.slider("Select video duration (seconds)", 2, 10, 4)
    if st.button("Generate Video"):
        if prompt:
            translated_prompt = translate_text(prompt, "en")
            st.write(f"Translated prompt: {translated_prompt}")
            with st.spinner("Generating video..."):
                video_frames = generate_video(translated_prompt, video_gen_model, duration)
            display_video(video_frames)
        else:
            st.warning("Please enter a prompt.")

# Image to Text
elif task == "Image to Text":
    uploaded_file = st.file_uploader("Choose an image to generate a caption", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_path = "uploaded_image.png"
        image.save(image_path)

        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption = predict_caption([image_path], captioning_model, feature_extractor, tokenizer)

            st.write(f"Generated Caption: {caption}")

            with st.spinner("Translating caption..."):
                translated_caption = translate_text(caption)

            st.write(f"Translated Caption: {translated_caption}")
