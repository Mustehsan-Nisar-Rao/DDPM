# 🎨 DDPM Face Generator

> Denoising Diffusion Probabilistic Model trained on CelebA-HQ dataset to generate realistic 128×128 human faces from pure Gaussian noise.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Project Structure](#project-structure)

---

## Overview

This project implements a **DDPM (Denoising Diffusion Probabilistic Model)** from scratch using PyTorch. The model learns to reverse a gradual noising process, starting from pure random noise and iteratively denoising it over 500 timesteps to generate realistic human face images.

**Key Highlights:**
- Trained on CelebA-HQ (high quality celebrity faces)
- Custom UNet architecture with time embeddings
- 50 epochs of training with best model checkpointing
- Streamlit web app for interactive image generation

---

## Model Architecture

The model uses a **UNet** backbone with residual blocks and sinusoidal time embeddings.

```
Input: Random Noise (1, 3, 128, 128)
        ↓
   TimeEmbedding (sinusoidal → MLP)
        ↓
   Encoder
   ├── Level 1: ResBlock × 2  →  64 channels   + Downsample
   ├── Level 2: ResBlock × 2  →  128 channels  + Downsample
   └── Level 3: ResBlock × 2  →  256 channels  (bottleneck)
        ↓
   Bottleneck: ResBlock × 2
        ↓
   Decoder (with skip connections)
   ├── Level 3: ResBlock × 2  →  256 channels  + Upsample
   ├── Level 2: ResBlock × 2  →  128 channels  + Upsample
   └── Level 1: ResBlock × 2  →  64 channels
        ↓
   Output: Generated Image (1, 3, 128, 128)
```

| Component | Details |
|-----------|---------|
| Base Channels | 64 |
| Channel Multipliers | (1, 2, 4) |
| Time Embedding Dim | 128 |
| Normalization | GroupNorm |
| Activation | GELU |
| Dropout | 0.1 |
| Total Parameters | ~10.8M |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | CelebA-HQ |
| Image Size | 128 × 128 |
| Timesteps | 500 |
| Beta Schedule | Linear (0.0001 → 0.02) |
| Epochs | 50 |
| Batch Size | ~16 |
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-2 |
| Mixed Precision | ✅ AMP (autocast) |
| Gradient Clipping | 1.0 |

### Training Loss Curve

| Epoch | Avg Loss |
|-------|---------|
| 1 | 0.047996 |
| 5 | 0.019356 |
| 10 | 0.018564 |
| 20 | 0.017335 |
| 35 | 0.016501 |
| **46** | **0.016364** ← Best |
| 50 | 0.016945 |

> **Best model saved at Epoch 46** with loss `0.016364`

---

## Results

| Metric | Value | Note |
|--------|-------|------|
| Best Training Loss | 0.016364 | Epoch 46 |
| PSNR | ~11 dB | Normal for generative models |
| SSIM | ~0.22 | Normal for generative models |

> **Note:** PSNR and SSIM are low by design — the model generates *new* faces, not reconstructions of existing images. For generative models, **FID (Fréchet Inception Distance)** is the standard evaluation metric.

### Noise Schedule (Forward Process)

```
t=0   →  Clean image
t=100 →  Slightly noisy
t=250 →  Half noisy
t=499 →  Pure Gaussian noise
```

---

## Installation

**Requirements:** Python 3.10

```bash
# 1. Clone the repo
git clone https://github.com/Mustehsan-Nisar-Rao/DDPM.git
cd DDPM

# 2. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit==1.35.0
torch==2.1.2
torchvision==0.16.2
numpy==1.26.4
Pillow==10.3.0
requests==2.31.0
```

---

## Usage

### Generate Images (Python)

```python
import torch

# Load model
model = UNet(in_channels=3, out_channels=3, base_channels=64,
             time_emb_dim=128, channel_mults=(1, 2, 4)).to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# Generate
generated = generate_images(num_images=4)
```

### Reconstruct from Image

```python
reconstruct_image('your_image.jpg', noise_level=250)
# noise_level: 0-499
# lower  → output closer to original
# higher → more creative / different output
```

---

## Streamlit App

An interactive web app that generates faces and shows intermediate denoising steps.

```bash
streamlit run app.py
```

### App Features

| Feature | Description |
|---------|-------------|
| 🎲 Random / Fixed Seed | Reproducible generation |
| 🔢 Intermediate Steps | Show 4–16 denoising snapshots |
| ⬇ Download | Save generated face as PNG |
| 🖥 Auto Device | Detects GPU / CPU automatically |
| ☁ Auto Download | Model weights fetched from GitHub on first run |

**Model weights are automatically downloaded from:**
```
https://github.com/Mustehsan-Nisar-Rao/DDPM/releases/tag/v1/best_model.pt
```

---

## Project Structure

```
DDPM/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── diffusion.ipynb         # Training notebook (Kaggle/Colab)
├── inference_correct.py    # Inference script
├── best_model.pt           # Trained weights (GitHub Release)
└── README.md               # This file
```

---

## How Diffusion Works

```
FORWARD PROCESS (Training):
Clean Image → Add noise gradually → Pure noise
x₀ ──────────────────────────────────────→ xₜ

REVERSE PROCESS (Inference):
Pure noise → Remove noise step by step → Clean Image
xₜ ──────────────────────────────────────→ x₀
```

The UNet learns to predict the noise `ε` added at each timestep `t`, allowing it to reverse the process during inference.

---

## References

- [DDPM Paper — Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [CelebA-HQ Dataset](https://github.com/tkarras/progressive_growing_of_gans)

---

<div align="center">
Made with PyTorch · Trained on Kaggle · Deployed with Streamlit
</div>
