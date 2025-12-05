import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Load models
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
)
pipe = pipe.to(device)
if device == "cuda":
    pipe.enable_attention_slicing()

print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print(f"Running on: {device}")

def generate_image(prompt, cfg_scale, num_steps, seed):
    """Generate image from text prompt."""
    
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    image = pipe(
        prompt,
        num_inference_steps=int(num_steps),
        guidance_scale=cfg_scale,
        generator=generator,
    ).images[0]
    
    # Compute CLIP score
    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_score = outputs.logits_per_image.item() / 100
    
    return image, f"CLIP Score: {clip_score:.4f}"

# Create Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label=" Enter Your Prompt",
            placeholder="A cat sitting on a beach at sunset...",
            info="Describe the image you want to generate. Be specific for better results!"
        ),
        gr.Slider(
            minimum=1,
            maximum=15,
            value=7.5,
            step=0.5,
            label=" CFG Scale (Guidance)",
            info="Controls how closely the image follows your prompt. Low (1-5): Creative/loose. Medium (7-8): Balanced. High (10+): Strict/literal."
        ),
        gr.Slider(
            minimum=10,
            maximum=75,
            value=30,
            step=5,
            label=" Inference Steps",
            info="More steps = higher quality but slower. 20-30: Fast drafts. 50+: High quality. 75: Best for detailed images."
        ),
        gr.Slider(
            minimum=1,
            maximum=9999,
            value=42,
            step=1,
            label=" Seed",
            info="Random seed for reproducibility. Same seed + same prompt = same image. Change to get variations!"
        ),
    ],
    outputs=[
        gr.Image(label=" Generated Image"),
        gr.Textbox(label=" CLIP Alignment Score (Higher = Better Match to Prompt)"),
    ],
    title=" Text-to-Image Generator",
    description="""
## Generate Images from Text using Stable Diffusion v1.5

**How to use:**
1. Enter a descriptive prompt
2. Adjust the sliders to control generation
3. Click Submit and wait for your image

**Tips for good prompts:**
- Be specific: "A golden retriever puppy" > "A dog"
- Add style: "...in watercolor style" or "...photorealistic"
- Add lighting: "...at sunset" or "...with dramatic lighting"

**Built for IE 7615 Deep Learning Course @ Northeastern University**
    """,
    examples=[
        ["A golden retriever playing in the snow, photorealistic", 7.5, 30, 42],
        ["A futuristic city skyline at night with neon lights", 7.5, 30, 123],
        ["A bowl of fresh fruits on a wooden table, soft natural lighting", 7.5, 30, 456],
        ["An astronaut riding a horse on Mars, digital art", 10, 50, 789],
        ["A cozy coffee shop interior, rainy day outside, warm lighting", 7.5, 50, 999],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

demo.launch()
