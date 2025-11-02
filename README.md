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
1. A small red bicycle parked next to a brick wall, late afternoon
2. Two golden retriever puppies playing in tall grass
3. A bustling street market at night with colourful neon lights
4. A wooden dining table with a bowl of of oranges, natural light
5. A surfer riding a gaint wave under a dramatic cloudy sky
