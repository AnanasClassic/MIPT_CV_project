"""
Visualize LoRA-trained model by generating integral images.
"""

import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
import matplotlib.pyplot as plt

from PIL import Image


@dataclass(frozen=True)
class MetaRow:
    tex: str
    image_path: str


def _round_to_multiple(value: int, multiple: int = 8) -> int:
    if multiple <= 0:
        return int(value)
    return int((int(value) + multiple - 1) // multiple * multiple)


def load_metadata(csv_path: str | Path) -> list[MetaRow]:
    csv_path = Path(csv_path)
    rows: list[MetaRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tex = (row.get("tex") or "").strip()
            image_path = (row.get("image_path") or "").strip()
            if tex and image_path:
                rows.append(MetaRow(tex=tex, image_path=image_path))
    return rows


def infer_median_aspect_ratio(rows: list[MetaRow], root_dir: str | Path) -> float:
    """
    Returns median (width / height) from images referenced by metadata.
    Falls back to 1.0 if sizes can't be read.
    """
    root_dir = Path(root_dir)
    ratios: list[float] = []
    for r in rows:
        p = root_dir / r.image_path
        if not p.exists():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
            if w > 0 and h > 0:
                ratios.append(w / h)
        except Exception:
            continue
    return statistics.median(ratios) if ratios else 1.0


def _maybe_load_tokenizers_from_lora(pipe: DiffusionPipeline, lora_root: Path) -> None:
    """
    If the LoRA directory contains saved tokenizers (e.g. after adding special tokens),
    load them into the pipeline and resize text encoder embeddings to match vocab size.
    """
    tok_path = lora_root / "tokenizer"
    if tok_path.exists() and hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
        pipe.tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
        if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "resize_token_embeddings"):
            pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    tok2_path = lora_root / "tokenizer_2"
    if tok2_path.exists() and hasattr(pipe, "tokenizer_2") and pipe.tokenizer_2 is not None:
        pipe.tokenizer_2 = AutoTokenizer.from_pretrained(str(tok2_path), local_files_only=True)
        if hasattr(pipe, "text_encoder_2") and hasattr(pipe.text_encoder_2, "resize_token_embeddings"):
            pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))


def generate_integrals(
    model_id="UnfilteredAI/NSFW-gen-v2",
    lora_path="/home/ananasclassic/projects/CV_project/lora_integrals/final",
    num_samples=6,
    output_dir="/home/ananasclassic/projects/CV_project/lora_integrals/samples",
    height=512,
    width=512,
    csv_path="/home/ananasclassic/projects/CV_project/data/boi_images/metadata.csv",
    dataset_root="/home/ananasclassic/projects/CV_project/data/boi_images",
    auto_wide=True,
):
    """Generate and visualize integral images using LoRA model."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if PeftModel is None:
        raise RuntimeError("`peft` is required to load these LoRA weights. Install with: pip install peft")
    
    print("Loading base model...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=torch.float32,
        safety_checker=None,
    )

    _maybe_load_tokenizers_from_lora(pipe, Path(lora_path))
    
    pipe.unet = pipe.unet.float()
    pipe.vae = pipe.vae.float()
    pipe.text_encoder = pipe.text_encoder.float()
    pipe.text_encoder_2 = pipe.text_encoder_2.float()
    
    print(f"Loading LoRA from {lora_path}...")
    lora_root = Path(lora_path)
    unet_path = lora_root / "unet"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(unet_path if unet_path.exists() else lora_root))
    text_path = lora_root / "text_encoder"
    if text_path.exists():
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, str(text_path))
    text2_path = lora_root / "text_encoder_2"
    if text2_path.exists():
        pipe.text_encoder_2 = PeftModel.from_pretrained(pipe.text_encoder_2, str(text2_path))
    
    pipe = pipe.to(device)
    pipe.unet.eval()
    
    meta_rows = load_metadata(csv_path)
    if not meta_rows:
        raise RuntimeError(f"No rows found in metadata: {csv_path}")

    if auto_wide and (height, width) == (320, 1024):
        ratio = infer_median_aspect_ratio(meta_rows, dataset_root)
        if ratio > 1.05:
            width = 1024
            height = max(256, min(768, _round_to_multiple(int(width / ratio), 8)))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating images...")
    import math

    cols = 3
    grid_rows = max(1, math.ceil(min(num_samples, 12) / cols))
    cell_w = 3
    cell_h = 3
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cell_w * cols, cell_h * grid_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    chosen = random.sample(meta_rows, k=min(num_samples, len(meta_rows)))

    for i, r in enumerate(chosen):
        tex = r.tex
        prompt = f"Draw an integral <integral>${tex}$"
        
        print(f"\n{i+1}. Generating: {tex}")
        
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=50,
                guidance_scale=4,
                height=height,
                width=width,
            ).images[0]
        
        image.save(output_dir / f"integral_{i+1}.png")
        
        axes[i].imshow(image)
        axes[i].set_title(f"${tex}$", fontsize=10)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "grid.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved grid to {output_dir / 'grid.png'}")
    plt.close()
    
    print("\nDone! Images saved to:", output_dir)


def compare_with_original(
    lora_path="/home/ananasclassic/projects/CV_project/lora_integrals/final",
    test_prompts=None,
    height=320,
    width=1024,
):
    """Compare original vs LoRA-trained model side by side."""
    if PeftModel is None:
        raise RuntimeError("`peft` is required to load these LoRA weights. Install with: pip install peft")
    
    if test_prompts is None:
        test_prompts = [
            r"\int x^2 dx",
            r"\int \frac{1}{x} dx",
            r"\int e^x dx",
            r"\int \sin(x) dx",
        ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "UnfilteredAI/NSFW-gen-v2"
    
    print("Loading base model...")
    pipe_base = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    
    print("Loading LoRA model...")
    pipe_lora = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=torch.float32,
        safety_checker=None,
    )
    pipe_lora.unet = pipe_lora.unet.float()
    lora_root = Path(lora_path)

    _maybe_load_tokenizers_from_lora(pipe_lora, lora_root)

    unet_path = lora_root / "unet"
    pipe_lora.unet = PeftModel.from_pretrained(
        pipe_lora.unet, str(unet_path if unet_path.exists() else lora_root)
    )
    text_path = lora_root / "text_encoder"
    if text_path.exists():
        pipe_lora.text_encoder = PeftModel.from_pretrained(pipe_lora.text_encoder, str(text_path))
    text2_path = lora_root / "text_encoder_2"
    if text2_path.exists():
        pipe_lora.text_encoder_2 = PeftModel.from_pretrained(pipe_lora.text_encoder_2, str(text2_path))
    pipe_lora = pipe_lora.to(device)
    
    fig, axes = plt.subplots(len(test_prompts), 2, figsize=(10, 5 * len(test_prompts)))
    
    for i, tex in enumerate(test_prompts):
        prompt = f"Draw an integral {tex}"
        
        print(f"\nGenerating: {tex}")
        
        with torch.no_grad():
            img_base = pipe_base(
                prompt, num_inference_steps=50, height=height, width=width
            ).images[0]
        
        with torch.no_grad():
            img_lora = pipe_lora(
                prompt, num_inference_steps=50, height=height, width=width
            ).images[0]
        
        axes[i, 0].imshow(img_base)
        axes[i, 0].set_title(f"Base Model\n${tex}$")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_lora)
        axes[i, 1].set_title(f"LoRA Model\n${tex}$")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    output_path = Path("/home/ananasclassic/projects/CV_project/lora_integrals/comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "compare"], default="generate")
    parser.add_argument("--lora-path", default="/home/ananasclassic/projects/CV_project/lora_integrals/final")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--csv-path", default="/home/ananasclassic/projects/CV_project/data/boi_images/metadata.csv")
    parser.add_argument("--dataset-root", default="/home/ananasclassic/projects/CV_project/data/boi_images")
    parser.add_argument("--no-auto-wide", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_integrals(
            lora_path=args.lora_path,
            num_samples=args.num_samples,
            height=args.height,
            width=args.width,
            csv_path=args.csv_path,
            dataset_root=args.dataset_root,
            auto_wide=not args.no_auto_wide,
        )
    elif args.mode == "compare":
        compare_with_original(lora_path=args.lora_path, height=args.height, width=args.width)
