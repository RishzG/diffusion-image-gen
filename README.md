# 🎨 Text-to-Image Generation using Stable Diffusion

**Course:** IE 7615 Deep Learning for AI  
**University:** Northeastern University  
**Date:** December 2024

---

## Overview

This project implements a text-to-image generation pipeline using **Stable Diffusion v1.5** conditioned on **CLIP text embeddings**. We evaluate the model using FID, CLIP Score, and Inception Score metrics on the COCO Captions dataset.

---

## 🚀 Demo

**Try it live:** [Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/text-to-image-demo)

---

## 📊 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| FID Score ↓ | 37.22 | Good (lower is better) |
| CLIP Score ↑ | 0.3129 ± 0.0287 | Good text-image alignment |
| Inception Score ↑ | 29.95 ± 1.77 | Very good diversity/quality |

**Dataset:** COCO Captions (5,000 image-caption pairs)  
**Generated Images:** 2,500  
**Model:** Stable Diffusion v1.5  
**Text Encoder:** CLIP ViT-B/32

---

## 🔬 Experiments

### CFG Scale Sensitivity
Controls how closely the image follows the prompt.

| CFG Scale | Effect |
|-----------|--------|
| 1-5 | Creative, loose interpretation |
| 7-8 | Balanced (recommended) |
| 10+ | Strict, literal, higher contrast |

![CFG Sensitivity](figures/m2_cfg_sensitivity.png)

### Inference Steps Sensitivity
Controls image quality and generation time.

| Steps | Effect |
|-------|--------|
| 10 | Insufficient quality |
| 20-30 | Good balance of speed/quality |
| 50 | High quality |
| 75 | Best for detailed/geometric images |

**Key Finding:** The "diminishing returns" argument is context-dependent. Complex geometric patterns (e.g., umbrellas) show visible improvement at 75 steps compared to 50.

![Steps Sensitivity](figures/m2_steps_sensitivity.png)

---

## 📁 Project Structure

```
text-to-image-generation/
├── README.md
├── notebooks/
│   └── text2image_project.ipynb    # Main Colab notebook
├── src/
│   └── app.py                      # Gradio demo application
├── results/
│   ├── m1_data_stats.csv
│   ├── m2_baseline_clip_scores.csv
│   ├── m3_fid_score.csv
│   ├── m3_clip_scores.csv
│   ├── m3_inception_score.csv
│   └── m3_final_summary.csv
├── figures/
│   ├── m1_*.png                    # Data exploration plots
│   ├── m2_*.png                    # Baseline & experiment plots
│   └── m3_*.png                    # Final evaluation plots
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Setup

### Requirements
```bash
pip install -r requirements.txt
```

### Run Demo Locally
```bash
cd src
python app.py
```

### Run in Google Colab
1. Open `notebooks/text2image_project.ipynb` in Google Colab
2. Set runtime to **GPU** (T4 or A100)
3. Run all cells

---

## 📈 Methodology

### Milestone 1: Data Preparation
- Downloaded COCO Captions dataset (5,000 validation images)
- Created image-caption pairs with filtering (5-50 words)
- Explored caption length distribution and common words

### Milestone 2: Baseline Model & Experiments
- Integrated Stable Diffusion v1.5 with CLIP text encoder
- Generated baseline images (100 samples)
- Conducted CFG sensitivity analysis (3.0, 5.0, 7.5, 10.0, 12.0)
- Conducted inference steps analysis (10, 20, 30, 50, 75)
- Computed initial CLIP scores

### Milestone 3: Large-Scale Evaluation
- Generated 2,500 images for valid FID computation
- Computed FID Score: 37.22
- Computed CLIP Score: 0.3129 ± 0.0287
- Computed Inception Score: 29.95 ± 1.77
- Built interactive Gradio demo

---

## 📝 Key Findings

1. **FID Score of 37.22** indicates good image quality comparable to real COCO images

2. **CLIP Score of 0.3129** shows reasonable text-image alignment
   - Best aligned: Clear, specific subjects (animals, people with objects)
   - Worst aligned: Abstract concepts, multiple small objects

3. **Inception Score of 29.95** demonstrates good diversity and quality

4. **CFG Scale 7.5** provides the best balance between creativity and prompt adherence

5. **50 inference steps** is optimal for most cases, but **75 steps** improves detailed/geometric images

---

## ⚠️ Ethical Considerations

- **Dataset Bias:** COCO dataset may contain biases that propagate to generated images
- **Misuse Potential:** Text-to-image models can generate misleading or harmful content
- **Environmental Impact:** Large-scale generation requires significant compute resources
- **Copyright:** Generated images may inadvertently replicate training data patterns

---

## 🔮 Future Work

- Fine-tune on domain-specific datasets
- Experiment with different schedulers (DDIM, DPM++)
- Add negative prompt support
- Implement style control conditioning
- Compare with newer models (SDXL, SD 3.0)

---

## 👥 Team Members

- [Name 1]
- [Name 2]
- [Name 3]

---

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) by CompVis/Stability AI
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [COCO Dataset](https://cocodataset.org/)

---

## 📄 License

MIT License

---

## 📚 References

1. Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
2. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
3. Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
