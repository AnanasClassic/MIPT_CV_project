"""
Generate a side-by-side comparison image (base vs LoRA) for a single prompt.

Saves to: samples/{prompt_prefix}.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _slug_prefix(text: str, max_len: int = 48) -> str:
    text = text.strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9а-яА-Я._ -]+", "", text)
    text = text.strip().replace(" ", "_")
    if not text:
        return "prompt"
    return text[:max_len].rstrip("._- ")


def _maybe_load_tokenizers_from_lora(pipe: DiffusionPipeline, lora_root: Path) -> None:
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


def _load_with_lora(model_id: str, lora_path: Path, device: str) -> DiffusionPipeline:
    if PeftModel is None:
        raise RuntimeError("`peft` is required to load these LoRA weights. Install with: pip install peft")

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        dtype=torch.float32,
        safety_checker=None,
    )
    _maybe_load_tokenizers_from_lora(pipe, lora_path)

    pipe.unet = pipe.unet.float()
    pipe.vae = pipe.vae.float()
    pipe.text_encoder = pipe.text_encoder.float()
    pipe.text_encoder_2 = pipe.text_encoder_2.float()

    unet_path = lora_path / "unet"
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(unet_path if unet_path.exists() else lora_path))

    te_path = lora_path / "text_encoder"
    if te_path.exists():
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, str(te_path))

    te2_path = lora_path / "text_encoder_2"
    if te2_path.exists():
        pipe.text_encoder_2 = PeftModel.from_pretrained(pipe.text_encoder_2, str(te2_path))

    pipe = pipe.to(device)
    pipe.unet.eval()
    return pipe


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt string")
    parser.add_argument("--model-id", default="UnfilteredAI/NSFW-gen-v2")
    parser.add_argument("--lora-path", default=str(Path(__file__).resolve().parent / "lora_integrals" / "final"))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "samples"))
    parser.add_argument("--title-base", default="Base")
    parser.add_argument("--title-lora", default="LoRA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt = args.prompt
    lora_root = Path(args.lora_path).expanduser()
    if not lora_root.is_absolute():
        lora_root = (Path.cwd() / lora_root).resolve()
    if not lora_root.exists():
        raise FileNotFoundError(
            f"LoRA path not found: {lora_root} (cwd={Path.cwd()}). "
            "Pass an absolute path or a path relative to your current working directory."
        )

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator(device=device)
    if args.seed:
        gen.manual_seed(int(args.seed))

    pipe_base = DiffusionPipeline.from_pretrained(
        args.model_id,
        dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    pipe_lora = _load_with_lora(args.model_id, lora_root, device)

    with torch.no_grad():
        img_base = pipe_base(
            prompt,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance),
            height=int(args.height),
            width=int(args.width),
            generator=gen,
        ).images[0]

    with torch.no_grad():
        img_lora = pipe_lora(
            prompt,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance),
            height=int(args.height),
            width=int(args.width),
            generator=gen,
        ).images[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_base)
    axes[0].set_title(args.title_base)
    axes[0].axis("off")

    axes[1].imshow(img_lora)
    axes[1].set_title(args.title_lora)
    axes[1].axis("off")

    fig.text(0.5, 0.02, prompt, ha="center", va="bottom", wrap=True, fontsize=10)
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    out_path = out_dir / f"{_slug_prefix(prompt)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
