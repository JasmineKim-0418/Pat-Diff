import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy.ma as ma
from skimage.morphology import erosion, disk

from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(1)
else:
    device = torch.device("cpu")


class InferenceDataset(Dataset):
    """Slimmed-down dataset for inference: load MRI/PET pairs and tokenized captions only."""

    def __init__(self, mri_dir, pet_dir, caption_path, tokenizer, size=(256, 256)):
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.size = size
        self.tokenizer = tokenizer
        self.data_info = []

        print(f"Loading captions from: {caption_path}")
        with open(caption_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Verifying file sets"):
            if "|" not in line:
                continue
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            mri_fname, caption, label_str = parts
            mri_path = os.path.join(mri_dir, mri_fname)
            pet_path = os.path.join(pet_dir, mri_fname.replace("_MRI_", "_PET_"))
            if os.path.exists(mri_path) and os.path.exists(pet_path):
                new_caption = caption.replace(".", f", diagnosed with {label_str.upper()}.")
                self.data_info.append({
                    "mri_fname": mri_fname,
                    "caption": new_caption,
                    "label": label_str.strip(),
                })
        print(f"Found {len(self.data_info)} valid data entries.")

    def __len__(self):
        return len(self.data_info)

    def _preprocess_mri(self, img_path):
        """Z-score normalize an MRI .npy slice into [-1, 1] (matches train.py)."""
        img = np.load(img_path).astype(np.float32)
        mean, std = np.mean(img), np.std(img)
        img = (img - mean) / std if std > 1e-6 else img - mean
        img = np.clip(img, -3.0, 3.0) / 3.0

        img = ((img + 1.0) / 2.0 * 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img_pil = Image.fromarray(img).convert("RGB").resize(self.size)
        img_final = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img_final).permute(2, 0, 1)

    def _preprocess_pet(self, img_path):
        """Min-Max normalize a PET .npy slice into [-1, 1] (matches train.py)."""
        img = np.load(img_path).astype(np.float32)
        if img.max() > img.min():
            img = 255 * (img - img.min()) / (img.max() - img.min())
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img_pil = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(self.size)
        img_final = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img_final).permute(2, 0, 1)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        mri_fname, caption, label = info["mri_fname"], info["caption"], info["label"]
        mri_path = os.path.join(self.mri_dir, mri_fname)
        pet_path = os.path.join(self.pet_dir, mri_fname.replace("_MRI_", "_PET_"))

        mri_tensor = self._preprocess_mri(mri_path)
        pet_tensor = self._preprocess_pet(pet_path)
        encoding = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        return {
            "mri": mri_tensor,
            "pet": pet_tensor,
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "filename": mri_fname,
            "label": label,
        }


def save_comparison(original_mri, original_pet, generated_pet, label, save_path):
    """Save a 3-panel figure: input MRI | ground-truth PET | generated PET."""

    def process_tensor(tensor):
        img = tensor.cpu().detach().squeeze().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img / 2 + 0.5).clip(0, 1)
        img = np.rot90(img, k=-1)
        return img

    original_mri, original_pet, generated_pet = map(
        process_tensor, [original_mri, original_pet, generated_pet]
    )
    hot_modified = plt.get_cmap('hot').copy()
    hot_modified.set_bad(color='black')
    hot_modified.set_under('black')
    vmin, vmax = 0.2, 0.8

    def apply_colormap_and_mask(data, is_generated=False):
        if data.ndim == 3:
            data = data.mean(axis=-1)
        if is_generated:
            threshold = np.percentile(data[data > 0], 15) if np.any(data > 0) else 0.01
            mask = data > threshold
            if mask.sum() > 0:
                mask = erosion(mask, disk(1))
        else:
            mask = data > vmin
        return ma.masked_where(~mask, data)

    pet_norm_gt = apply_colormap_and_mask(original_pet)
    gen_gray_norm = apply_colormap_and_mask(generated_pet, is_generated=True)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original_mri)
    axs[0].set_title("Input MRI", fontsize=16, weight='bold')
    axs[0].axis("off")

    axs[1].imshow(pet_norm_gt, cmap=hot_modified, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Ground Truth PET ({label})", fontsize=16, weight='bold')
    axs[1].axis("off")

    axs[2].imshow(gen_gray_norm, cmap=hot_modified, vmin=vmin, vmax=vmax)
    axs[2].set_title("Generated PET", fontsize=16, weight='bold')
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def eta_from_t(t_tensor, total_timesteps):
    """Linear schedule per the paper (eta_0 ~= 0, eta_T ~= 1)."""
    return (t_tensor.float() / (total_timesteps - 1)).view(-1, 1, 1, 1)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading models from: {args.model_dir}")
    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.model_dir, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.model_dir, "text_encoder")).to(device)
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.model_dir, "unet")).to(device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    print("All models loaded successfully.")

    test_dataset = InferenceDataset(
        mri_dir=os.path.join(args.data_dir, "mri"),
        pet_dir=os.path.join(args.data_dir, "pet"),
        caption_path=args.caption_path,
        tokenizer=tokenizer,
        size=(args.image_size, args.image_size),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Test dataloader ready.")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_metric = MeanSquaredError().to(device)

    psnr_scores_by_label = {}
    ssim_scores_by_label = {}
    mse_scores_by_label = {}

    unet.eval()
    text_encoder.eval()

    for batch in tqdm(test_dataloader, desc="Generating images and computing metrics"):
        with torch.no_grad():
            source_mri = batch["mri"].to(device)
            real_pet = batch["pet"].to(device)
            filenames = [f.replace('_MRI.npy', '') for f in batch["filename"]]
            labels = batch["label"]
            batch_size = source_mri.shape[0]

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cond_context = text_encoder(input_ids, attention_mask=attention_mask)[0]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length",
                max_length=tokenizer.model_max_length, return_tensors="pt",
            )
            uncond_context = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=uncond_input.attention_mask.to(device),
            )[0]

            # Manual reverse process in pixel space.
            total_timesteps = noise_scheduler.config.num_train_timesteps  # 1000 (DDPM)
            timesteps = torch.linspace(total_timesteps - 1, 0, args.num_inference_steps).long().to(device)

            # Start from a noisy MRI, then denoise toward PET.
            t0 = timesteps[0]
            alpha_bar_t0 = alphas_cumprod[t0].view(1, 1, 1, 1)
            z0 = torch.randn_like(source_mri)
            images = source_mri + (1.0 - alpha_bar_t0).sqrt() * z0 * args.noise_strength

            # Denoising loop (paper Eq. (3)).
            for i, t_val in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
                t_cur = torch.full((batch_size,), t_val, device=device, dtype=torch.long)

                # CFG forward pass: concatenate uncond + cond batches.
                image_model_input = torch.cat([images] * 2, dim=0)
                t_input = torch.cat([t_cur] * 2, dim=0)
                combined_context = torch.cat([uncond_context, cond_context], dim=0)

                x0_pred = unet(image_model_input, t_input, encoder_hidden_states=combined_context).sample
                x0_u, x0_c = x0_pred.chunk(2, dim=0)
                x0_hat = x0_u + args.guidance_scale * (x0_c - x0_u)

                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1]
                    eta_t = eta_from_t(t_cur, total_timesteps)
                    eta_tnext = eta_from_t(torch.full_like(t_cur, t_next), total_timesteps)
                    images = (eta_tnext / eta_t) * images + ((eta_t - eta_tnext) / eta_t) * x0_hat
                else:
                    images = x0_hat

                # Log residual was learned inside the ROI, so the output mixes log-space values.
                # Strip the log via exp, subtract the 1.0 added during training, then clamp.
                pixel_space_images = torch.exp(images) - 1.0
                generated_images = pixel_space_images.clamp(-1, 1)

                generated_images_01 = (generated_images + 1) / 2
                real_pet_01 = (real_pet + 1) / 2

                for i in range(batch_size):
                    gen_img = generated_images_01[i].unsqueeze(0)
                    gt_img = real_pet_01[i].unsqueeze(0)
                    label = labels[i]

                    psnr_score = psnr_metric(gen_img, gt_img).item()
                    ssim_score = ssim_metric(gen_img, gt_img).item()
                    mse_score = mse_metric(gen_img, gt_img).item()

                    if label not in psnr_scores_by_label:
                        psnr_scores_by_label[label] = []
                        ssim_scores_by_label[label] = []
                        mse_scores_by_label[label] = []
                    psnr_scores_by_label[label].append(psnr_score)
                    ssim_scores_by_label[label].append(ssim_score)
                    mse_scores_by_label[label].append(mse_score)

                    filename_base = filenames[i]
                    output_path = os.path.join(
                        args.output_dir,
                        f"{label}_{filename_base}_comparison_g{args.guidance_scale}.png",
                    )
                    save_comparison(source_mri[i], real_pet[i], generated_images[i], label, output_path)

    # --- Final report ---
    print("\n" + "=" * 50)
    print("Inference and evaluation complete.")
    print(f"Results are saved in: {args.output_dir}")
    print("=" * 50)

    all_psnr, all_ssim, all_mse = [], [], []
    non_ad_psnr, non_ad_ssim, non_ad_mse = [], [], []

    for label in sorted(psnr_scores_by_label.keys()):
        psnr_scores = np.array(psnr_scores_by_label[label])
        ssim_scores = np.array(ssim_scores_by_label[label])
        mse_scores = np.array(mse_scores_by_label[label])

        all_psnr.extend(psnr_scores)
        all_ssim.extend(ssim_scores)
        all_mse.extend(mse_scores)

        if label != 'AD':
            non_ad_psnr.extend(psnr_scores)
            non_ad_ssim.extend(ssim_scores)
            non_ad_mse.extend(mse_scores)

        print(f"--- Label: {label} ({len(psnr_scores)} samples) ---")
        print(f"  PSNR: {psnr_scores.mean():.4f} ± {psnr_scores.std():.4f}")
        print(f"  SSIM: {ssim_scores.mean():.4f} ± {ssim_scores.std():.4f}")
        print(f"  MSE:  {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")

    print("-" * 50)
    print(f"--- Performance (excluding AD) ({len(non_ad_psnr)} samples) ---")
    print(f"  Average PSNR: {np.mean(non_ad_psnr):.4f} ± {np.std(non_ad_psnr):.4f}")
    print(f"  Average SSIM: {np.mean(non_ad_ssim):.4f} ± {np.std(non_ad_ssim):.4f}")
    print(f"  Average MSE:  {np.mean(non_ad_mse):.4f} ± {np.std(non_ad_mse):.4f}")

    print("-" * 50)
    print("--- Overall Performance ---")
    print(f"Average PSNR: {np.mean(all_psnr):.4f} ± {np.std(all_psnr):.4f}")
    print(f"Average SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"Average MSE:  {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for the PaT-Diff residual diffusion model.")
    parser.add_argument("--model_dir", type=str,
                        default="./outputs/log_rd_on_roi_lam0.2/final_model_for_inference",
                        help="Directory of trained models.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the test dataset (containing mri and pet subfolders).")
    parser.add_argument("--caption_path", type=str, required=True,
                        help="Path to the caption file for the test set.")
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                        help="Directory to save the generated images and comparisons.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Guidance scale for classifier-free guidance.")
    parser.add_argument("--noise_strength", type=float, default=0.1,
                        help="Forward noise std scale (must match training).")
    args = parser.parse_args()
    main(args)
