import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps, ImageChops
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline, DDPMScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chk(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        bad = t[~torch.isfinite(t)]
        mn = float(t.min().detach().cpu()) if t.numel() else float("nan")
        mx = float(t.max().detach().cpu()) if t.numel() else float("nan")
        print(f"[NaN/Inf] {name}: dtype={t.dtype} shape={tuple(t.shape)} min={mn} max={mx} bad={bad.numel()}")
        raise RuntimeError(f"Non-finite in {name}")


def _compute_snr(noise_scheduler: DDPMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha = alphas_cumprod[timesteps]
    snr = alpha / (1.0 - alpha)
    return torch.clamp(snr, min=1e-6)


def _get_pred_type(noise_scheduler: DDPMScheduler) -> str:
    return getattr(noise_scheduler.config, "prediction_type", "epsilon")


def _get_target(
    noise_scheduler: DDPMScheduler,
    pred_type: str,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        if hasattr(noise_scheduler, "get_velocity"):
            return noise_scheduler.get_velocity(latents, noise, timesteps)
        raise RuntimeError("Scheduler is v_prediction but has no get_velocity().")
    raise ValueError(f"Unknown prediction_type: {pred_type}")


def _predict_x0(
    noise_scheduler: DDPMScheduler,
    pred_type: str,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    model_pred: torch.Tensor,
) -> torch.Tensor:
    dtype = noisy_latents.dtype
    x_t = noisy_latents.float()
    p = model_pred.float()

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=torch.float32)
    a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)

    if pred_type == "epsilon":
        x0 = (x_t - torch.sqrt(1.0 - a_t) * p) / torch.sqrt(a_t)
    elif pred_type == "v_prediction":
        x0 = torch.sqrt(a_t) * x_t - torch.sqrt(1.0 - a_t) * p
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

    return x0.to(dtype)


def _sobel_edges(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 3:
        gray = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]
    else:
        gray = images

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class CropToContent:
    """
    Crop by bbox of non-background pixels (assumes near-white background).
    """

    def __init__(self, margin: int = 12, bg_rgb: Tuple[int, int, int] = (255, 255, 255), min_box: int = 8):
        self.margin = int(margin)
        self.bg_rgb = tuple(bg_rgb)
        self.min_box = int(min_box)

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        bg = Image.new("RGB", image.size, self.bg_rgb)
        diff = ImageChops.difference(image, bg)
        diff = ImageOps.autocontrast(diff)

        bbox = diff.getbbox()
        if bbox is None:
            return image

        x0, y0, x1, y1 = bbox
        if (x1 - x0) < self.min_box or (y1 - y0) < self.min_box:
            return image

        x0 = max(0, x0 - self.margin)
        y0 = max(0, y0 - self.margin)
        x1 = min(image.size[0], x1 + self.margin)
        y1 = min(image.size[1], y1 + self.margin)

        return image.crop((x0, y0, x1, y1))


class ResizePadToSquare:
    def __init__(self, size: int, fill=(255, 255, 255), interpolation=Image.BICUBIC):
        self.size = int(size)
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.contain(image, (self.size, self.size), method=self.interpolation)
        pad_w = self.size - image.size[0]
        pad_h = self.size - image.size[1]
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - (pad_w // 2),
            pad_h - (pad_h // 2),
        )
        return ImageOps.expand(image, border=padding, fill=self.fill)


class IntegralDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        image_size: int = 512,
        indices: Optional[list] = None,
        crop_to_content: bool = True,
        crop_margin: int = 12,
    ):
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.image_size = int(image_size)

        self.df = pd.read_csv(self.csv_path)
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)

        pre = []
        if crop_to_content:
            pre.append(CropToContent(margin=crop_margin))
        pre.append(ResizePadToSquare(self.image_size, fill=(255, 255, 255)))

        self.transforms = transforms.Compose(
            pre
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tex = str(row["tex"])
        image_path = row["image_path"]

        full_image_path = self.root_dir / image_path
        try:
            image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"Warning loading {full_image_path}: {e}")
            image = Image.new("RGB", (self.image_size, self.image_size), color="white")

        image = self.transforms(image)
        prompt = f"Draw an integral {tex}"

        return {
            "image": image,
            "prompt": prompt,
            "tex": tex,
            "original_size": (self.image_size, self.image_size),
        }


def main():
    model_id = "UnfilteredAI/NSFW-gen-v2"
    dataset_path = "/home/ananasclassic/projects/CV_project/data/synt/metadata.csv"
    dataset_root = "/home/ananasclassic/projects/CV_project/data/synt"
    output_dir = "/home/ananasclassic/projects/CV_project/lora_integrals"

    use_tensorboard = True
    tensorboard_dir = str(Path(output_dir) / "runs")
    tb_log_every_n_steps = 10

    seed = 42

    image_size = 512
    batch_size = 4
    num_epochs = 10
    learning_rate = 4e-5
    val_split = 0.1

    latent_x0_loss_weight = 0.0
    image_recon_loss_weight = 0.05
    image_recon_every_n_steps = 1
    edge_loss_weight = 0.05
    x0_image_t_threshold = 400

    min_snr_gamma = 5.0

    precision = "fp32"
    grad_clip = 1.0

    val_seed_base = 123456
    val_batches = 2

    train_text_encoders = False

    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        precision = "fp32"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if precision == "bf16" and device.type == "cuda":
        weight_dtype = torch.bfloat16
        use_autocast = True
    else:
        weight_dtype = torch.float32
        use_autocast = False

    print(f"Device: {device} | precision={precision} | weight_dtype={weight_dtype} | autocast={use_autocast}")

    print("Loading dataset...")
    full_dataset = IntegralDataset(dataset_path, dataset_root, image_size=image_size, crop_to_content=True, crop_margin=12)

    n_samples = len(full_dataset)
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_dataset = IntegralDataset(dataset_path, dataset_root, image_size=image_size, indices=train_indices, crop_to_content=True, crop_margin=12)
    val_dataset = IntegralDataset(dataset_path, dataset_root, image_size=image_size, indices=val_indices, crop_to_content=True, crop_margin=12)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        prompts = [item["prompt"] for item in batch]
        tex = [item["tex"] for item in batch]
        original_sizes = [item["original_size"] for item in batch]
        return {"images": images, "prompts": prompts, "tex": tex, "original_sizes": original_sizes}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print("Loading pipeline (fp32 base for stability)...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)

    pipe.unet.to(dtype=weight_dtype)

    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    pipe.vae.to(dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()

    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = noise_scheduler
    pred_type = _get_pred_type(noise_scheduler)
    print(f"Scheduler prediction_type: {pred_type}")

    print("Setting up UNet LoRA...")
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("Need peft installed: pip install peft")

    unet_lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["to_k", "to_v", "to_q", "to_out.0"],
    )
    pipe.unet = get_peft_model(pipe.unet, unet_lora)
    pipe.unet.train()

    optim_groups = [{"params": [p for p in pipe.unet.parameters() if p.requires_grad], "lr": learning_rate}]
    optimizer = torch.optim.AdamW(optim_groups)
    trainable_params = [p for p in optim_groups[0]["params"]]

    if use_autocast:
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext

    global_step = 0

    tb = None
    if use_tensorboard:
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard logging requested but not available. Install with: pip install tensorboard")
        tb = SummaryWriter(log_dir=tensorboard_dir)

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_img_recon = 0.0
        epoch_latent_x0 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step_idx, batch in enumerate(pbar):
            images = batch["images"].to(device=device, dtype=torch.float32)
            prompts = batch["prompts"]
            original_sizes = batch["original_sizes"]

            chk("images", images)

            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
            chk("latents", latents)

            with torch.no_grad():
                text_inputs = pipe.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                text_inputs_2 = pipe.tokenizer_2(
                    prompts,
                    padding="max_length",
                    max_length=pipe.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                out1 = pipe.text_encoder(text_inputs.input_ids, output_hidden_states=True)
                out2 = pipe.text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True)

                prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
                pooled_prompt_embeds = getattr(out2, "text_embeds", None)
                if pooled_prompt_embeds is None:
                    pooled_prompt_embeds = out2[0]

            chk("prompt_embeds", prompt_embeds)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            chk("noisy_latents", noisy_latents)

            add_time_ids = torch.tensor(
                [[h, w, 0, 0, image_size, image_size] for (h, w) in original_sizes],
                device=device,
                dtype=prompt_embeds.dtype,
            )
            added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                model_pred = pipe.unet(
                    noisy_latents.to(dtype=weight_dtype),
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample
            chk("model_pred", model_pred)

            target = _get_target(noise_scheduler, pred_type, latents, noise, timesteps)
            chk("target", target)

            mse_per = (model_pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3))

            if min_snr_gamma > 0.0:
                snr = _compute_snr(noise_scheduler, timesteps)
                gamma = torch.full_like(snr, float(min_snr_gamma))
                if pred_type == "v_prediction":
                    weights = torch.minimum(snr, gamma) / (snr + 1.0)
                else:
                    weights = torch.minimum(snr, gamma) / snr
                loss_diffusion = (weights * mse_per).mean()
            else:
                loss_diffusion = mse_per.mean()

            loss = loss_diffusion

            do_image_decode = (
                (image_recon_loss_weight > 0.0 or edge_loss_weight > 0.0) and (step_idx % int(image_recon_every_n_steps) == 0)
            )
            need_x0 = (latent_x0_loss_weight > 0.0) or do_image_decode
            pred_x0 = _predict_x0(noise_scheduler, pred_type, noisy_latents, timesteps, model_pred) if need_x0 else None

            if latent_x0_loss_weight > 0.0 and pred_x0 is not None:
                t_mask = timesteps < int(x0_image_t_threshold)
                if bool(t_mask.any()):
                    loss_latent_x0 = F.mse_loss(pred_x0[t_mask].float(), latents[t_mask].float())
                    loss = loss + float(latent_x0_loss_weight) * loss_latent_x0
                    epoch_latent_x0 += float(loss_latent_x0.detach().item())

            if do_image_decode and pred_x0 is not None:
                t_mask = timesteps < int(x0_image_t_threshold)
                if bool(t_mask.any()):
                    with torch.autocast(device_type="cuda", enabled=False):
                        decoded = pipe.vae.decode((pred_x0[t_mask].float()) / pipe.vae.config.scaling_factor).sample
                    gt = images[t_mask].float()
                    chk("decoded", decoded)

                    if image_recon_loss_weight > 0.0:
                        loss_img_recon = F.l1_loss(decoded.float(), gt)
                        loss = loss + float(image_recon_loss_weight) * loss_img_recon
                        epoch_img_recon += float(loss_img_recon.detach().item())

                    if edge_loss_weight > 0.0:
                        loss_edge = F.l1_loss(_sobel_edges(decoded.float()), _sobel_edges(gt))
                        loss = loss + float(edge_loss_weight) * loss_edge

            chk("loss", loss)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip))
            optimizer.step()

            global_step += 1
            epoch_loss += float(loss.detach().item())

            if tb is not None and (global_step % int(tb_log_every_n_steps) == 0):
                tb.add_scalar("train/loss_total", float(loss.detach().item()), global_step)
                tb.add_scalar("train/loss_diffusion", float(loss_diffusion.detach().item()), global_step)
                tb.add_scalar("train/timestep_mean", float(timesteps.float().mean().detach().item()), global_step)
                if min_snr_gamma > 0.0:
                    tb.add_scalar("train/snr_mean", float(snr.float().mean().detach().item()), global_step)
                tb.add_scalar("optim/grad_norm", float(grad_norm.detach().item()) if hasattr(grad_norm, "detach") else float(grad_norm), global_step)

            denom_img = max(1, (step_idx // max(1, image_recon_every_n_steps)) + 1)
            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach().item()):.4f}",
                    "x0": f"{epoch_latent_x0 / max(1, step_idx + 1):.4f}",
                    "img": f"{(epoch_img_recon / denom_img) if image_recon_loss_weight > 0.0 else 0.0:.4f}",
                }
            )

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        if tb is not None:
            tb.add_scalar("train/epoch_avg_loss", float(avg_loss), epoch + 1)

        pipe.unet.eval()
        with torch.no_grad():
            val_seen = 0
            val_l1_sum = 0.0
            for vb_idx, vb in enumerate(val_loader):
                if val_seen >= int(val_batches):
                    break
                g = torch.Generator(device=device).manual_seed(int(val_seed_base + vb_idx))

                pred01 = pipe(
                    vb["prompts"],
                    num_inference_steps=25,
                    guidance_scale=2.0,
                    height=image_size,
                    width=image_size,
                    output_type="pt",
                    generator=g,
                ).images.to(device=device, dtype=torch.float32)

                pred = pred01 * 2.0 - 1.0
                gt = vb["images"].to(device=device, dtype=torch.float32)
                val_l1_sum += float(F.l1_loss(pred, gt).detach().item())
                val_seen += 1

            if val_seen:
                val_l1 = float(val_l1_sum / val_seen)
                print(f"Epoch {epoch+1} - Val image L1: {val_l1:.4f}")
                if tb is not None:
                    tb.add_scalar("val/image_l1", val_l1, epoch + 1)

        pipe.unet.train()

        save_path = Path(output_dir) / f"epoch_{epoch+1}"
        save_path.mkdir(parents=True, exist_ok=True)

        (save_path / "unet").mkdir(parents=True, exist_ok=True)
        pipe.unet.save_pretrained(save_path / "unet")

        print(f"Saved checkpoint: {save_path}")

    final_path = Path(output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    (final_path / "unet").mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(final_path / "unet")

    if tb is not None:
        tb.flush()
        tb.close()

    print(f"Training complete! Adapters saved to {final_path}")


if __name__ == "__main__":
    main()
