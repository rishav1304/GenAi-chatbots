# MultiVision Ai
MultiVision Ai is a  multi-modal AI-powered application that combines the capabilities of text-to-image, text-to-video, and image-to-text models to generate creative content and captions based on user inputs. This project leverages cutting-edge machine learning models from **Hugging Face** and utilizes **Streamlit** for an interactive user interface.

---

## Features

1. **Text-to-Image Generation**  
   Generate high-quality images from user-provided text prompts using **Stable Diffusion** (stabilityai/stable-diffusion-2).

2. **Text-to-Video Generation**  
   Create short dynamic videos from text descriptions using **Damo-ViLab Text-to-Video Model** (damo-vilab/text-to-video-ms-1.7b).

3. **Image Captioning (Image-to-Text)**  
   Extract meaningful captions from uploaded images using **Vision Encoder Decoder Model** (nlpconnect/vit-gpt2-image-captioning).

4. **Multi-language Translation**  
   Supports multi-language input for text prompts, with translation handled by the **Google Translate API**.

---

## Models Overview

### 1. Stable Diffusion
- **Model ID:** `stabilityai/stable-diffusion-2`
- **Purpose:** Text-to-Image Generation  
- **Why Chosen:**  
  - State-of-the-art diffusion model for creating high-quality, customizable images.
  - Offers control over guidance scale, inference steps, and seed-based randomization.

### 2. Damo-ViLab Text-to-Video
- **Model ID:** `damo-vilab/text-to-video-ms-1.7b`
- **Purpose:** Text-to-Video Generation  
- **Why Chosen:**  
  - Top-performing model for generating short videos from text prompts.
  - Combines transformer architecture and diffusion techniques for seamless video synthesis.

### 3. Vision Encoder Decoder Model
- **Model ID:** `nlpconnect/vit-gpt2-image-captioning`
- **Purpose:** Image-to-Text Captioning  
- **Why Chosen:**  
  - Efficient and accurate in producing human-like captions for images.
  - Combines a vision transformer (ViT) with a GPT-2-based text generator for superior results.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ReImagination.git
   cd ReImagination
