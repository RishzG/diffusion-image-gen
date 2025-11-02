
# Milestone 1 — Text-to-Image (CLIP + Diffusers)

## Quick Start
```bash
# 1) Create and activate environment (example: conda)
conda create -n t2i python=3.11 -y
conda activate t2i

# 2) Install deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers diffusers accelerate safetensors pillow sentencepiece

# 3) Build pairs.csv (COCO example)
python preprocess_coco_subset.py   --images_root /data/coco/train2017   --captions_json /data/coco/annotations/captions_train2017.json   --out_csv data/coco_small/pairs.csv   --max_pairs 5000

# 4) Open the baseline notebook
jupyter lab baseline_text_to_embedding.ipynb
```

## Sample 5 Prompts (example)
1. a cozy cabin in the woods during golden hour, 35mm photo, high detail
2. a ceramic mug shaped like a cat, studio lighting, product shot
3. a futuristic city skyline at night with neon lights, wide angle
4. a plate of sushi arranged as a smiley face, overhead shot
5. a bicycle parked beside a sunflower field, pastel color palette

Put generated images under `outputs/` and commit to the repo.
