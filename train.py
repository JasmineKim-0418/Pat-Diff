import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import nibabel as nib
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================
# Residual: MRI - PET, ROI map as cross-attention condition
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda:2")
    torch.cuda.set_device(2)
else:
    device = torch.device("cpu")


class Config:
    image_size = 256
    batch_size = 8
    epochs = 150
    lr = 1e-5
    cross_attention_dim = 768
    noise_strength = 0.1
    roi_map_path = "/DataCommon2/ksoh/ws/ADNI/MCICN_vs_sMCIsCN_common_pvalue_05_01_001_map.nii.gz"
    clip_finetune_strategy = "none"
    residual_mode = "log_rd_on_roi"

    use_roi_weighted_loss = True
    lambda_roi_loss = 0.2
    lambda_log_loss = 0.2


class MultiCondDataset(Dataset):
    def __init__(self, mri_dir, pet_dir, roi_map_path, caption_path, tokenizer,
                 size=(256, 256), train_mode=True):
        self.mri_dir = mri_dir
        self.pet_dir = pet_dir
        self.size = size
        self.tokenizer = tokenizer
        self.data = []
        self.train_mode = train_mode

        self.label_map = {'CN': 0, 'SMC': 1, 'MCI': 2, 'EMCI': 3, 'LMCI': 4, 'AD': 5}

        # Load the common ROI map once and keep it in memory.
        print(f"Loading ROI map from: {roi_map_path}")
        self.common_roi_map_tensor = self._preprocess_common_map(roi_map_path)

        print(f"Loading captions from: {caption_path}")
        with open(caption_path, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Verifying file sets"):
            if "|" not in line:
                continue
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue

            mri_fname, original_caption, label_str = parts
            mri_path = os.path.join(mri_dir, mri_fname)
            pet_path = os.path.join(pet_dir, mri_fname.replace("_MRI_", "_PET_"))

            if os.path.exists(mri_path) and os.path.exists(pet_path):
                if label_str.upper() in self.label_map:
                    label = self.label_map[label_str.upper()]
                    new_caption = original_caption.replace(".", f", diagnosed with {label_str.upper()}.")
                    self.data.append((mri_path, pet_path, new_caption, label, label_str.upper()))
            else:
                if not os.path.exists(mri_path):
                    print(f"Missing: {mri_path}")
                if not os.path.exists(pet_path):
                    print(f"Missing: {pet_path}")

        if len(self.data) == 0:
            raise RuntimeError("No valid data sets found.")
        print(f"Found {len(self.data)} valid data sets.")

    def __len__(self):
        return len(self.data)

    def _preprocess_common_map(self, map_path):
        """Load a .nii.gz ROI map, take the central axial slice, normalize to [-1, 1]."""
        nii_img = nib.load(map_path)
        img = np.squeeze(nii_img.get_fdata().astype(np.float32))

        if img.ndim == 3:
            img = img[:, :, img.shape[-1] // 2]
        elif img.ndim != 2:
            raise ValueError(f"Unexpected image dimension: {img.ndim}, shape: {img.shape}")

        if img.max() > img.min():
            img = 255 * (img - img.min()) / (img.max() - img.min())
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        img_pil = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(self.size)
        img_final = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img_final).permute(2, 0, 1)

    def _preprocess_mri(self, img_path):
        """Z-score normalize an MRI .npy slice into [-1, 1] using a 3-sigma clip."""
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
        """Min-Max normalize a PET .npy slice into [-1, 1]."""
        img = np.load(img_path).astype(np.float32)
        if img.max() > img.min():
            img = 255 * (img - img.min()) / (img.max() - img.min())
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img_pil = Image.fromarray(img.astype(np.uint8)).convert("RGB").resize(self.size)
        img_final = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(img_final).permute(2, 0, 1)

    def __getitem__(self, idx):
        mri_path, pet_path, caption, label, _ = self.data[idx]
        mri_tensor = self._preprocess_mri(mri_path)
        pet_tensor = self._preprocess_pet(pet_path)
        encoding = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        )
        return {
            "mri": mri_tensor,
            "pet": pet_tensor,
            "roi_map": self.common_roi_map_tensor,
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def compute_residual(source_mri, target_pet, roi_map, eps=1e-8):
    """
    log_rd_on_roi: hybrid residual that uses log RD inside the ROI
    (sensitive to fine-grained, nonlinear intensity variations) and pixel RD outside
    (preserves the global intensity distribution). U-Net's prediction target is PET.
    """
    rd_pixel = source_mri - target_pet

    mri_pos = source_mri + 1.0 + eps
    pet_pos = target_pet + 1.0 + eps
    rd_log = torch.clamp(torch.log(mri_pos) - torch.log(pet_pos), -3.0, 3.0)

    roi_mask = (roi_map + 1.0) / 2.0  # [0, 1] mask
    return (1 - roi_mask) * rd_pixel + roi_mask * rd_log


def pwr_loss(predicted, target, roi_map, lambda_roi=0.2, lambda_mae=0.5, use_roi_weight=True):
    """Pathology-Weighted Reconstruction loss: ROI-weighted L1 + L2."""
    abs_err = torch.abs(predicted - target)
    sq_err = abs_err ** 2

    if use_roi_weight:
        roi_mask = (roi_map + 1.0) / 2.0
        weight_map = 1.0 + lambda_roi * roi_mask
        mse = torch.mean(sq_err * weight_map)
        mae = torch.mean(abs_err * weight_map)
    else:
        mse = torch.mean(sq_err)
        mae = torch.mean(abs_err)
    return (1 - lambda_mae) * mse + lambda_mae * mae


def save_debug_visualization(source_mri, target_images, roi_map, residual_e0, output_dir, epoch, step):
    """Save a 5-panel debug figure: MRI, PET, RD, log-RD, ROI overlay."""
    import matplotlib.pyplot as plt
    os.makedirs(f"{output_dir}/debug_vis", exist_ok=True)

    idx = 0
    mri_img = source_mri[idx].detach().cpu().numpy().transpose(1, 2, 0)
    pet_img = target_images[idx].detach().cpu().numpy().transpose(1, 2, 0)
    rd_img = residual_e0[idx].detach().cpu().numpy().transpose(1, 2, 0)
    roi_img = ((roi_map[idx] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0)

    log_rd_img = np.log(mri_img + 1.0 + 1e-8) - np.log(pet_img + 1.0 + 1e-8)
    log_rd_img = np.clip(log_rd_img, -3.0, 3.0)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 5, 1); plt.imshow((mri_img + 1) / 2); plt.title("MRI")
    plt.subplot(1, 5, 2); plt.imshow((pet_img + 1) / 2); plt.title("PET")
    plt.subplot(1, 5, 3); plt.imshow((rd_img - rd_img.min()) / (rd_img.max() - rd_img.min()), cmap="bwr"); plt.title("Residual (RD)")
    plt.subplot(1, 5, 4); plt.imshow((log_rd_img - log_rd_img.min()) / (log_rd_img.max() - log_rd_img.min()), cmap="bwr"); plt.title("Log Residual")
    plt.subplot(1, 5, 5); plt.imshow((mri_img + 1) / 2, cmap="gray"); plt.imshow(roi_img[:, :, 0], cmap="hot", alpha=0.4); plt.title("ROI Overlay")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/debug_vis/epoch{epoch:03d}_step{step:05d}.png", dpi=150)
    plt.close()


if __name__ == '__main__':
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    clip_model_name = "openai/clip-vit-large-patch14"
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)

    unet = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=config.cross_attention_dim,
    ).to(device)

    if config.clip_finetune_strategy == "full":
        text_encoder.requires_grad_(True)
        print("Text Encoder strategy: full fine-tuning.")
    elif config.clip_finetune_strategy == "last_layer":
        text_encoder.requires_grad_(False)
        for param in text_encoder.text_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        print("Text Encoder strategy: fine-tune the last transformer layer.")
    else:
        text_encoder.requires_grad_(False)
        print("Text Encoder strategy: frozen.")

    # Differential learning rates: text encoder uses a much lower LR than the U-Net.
    lr_unet = config.lr
    lr_text_encoder = 1e-6
    optimizer_grouped_parameters = [{"params": unet.parameters(), "lr": lr_unet}]
    if config.clip_finetune_strategy != "none":
        trainable_text_encoder_params = list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
        optimizer_grouped_parameters.append({"params": trainable_text_encoder_params, "lr": lr_text_encoder})
        print(f"Optimizer LRs: U-Net={lr_unet}, Text Encoder={lr_text_encoder}")
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    train_dataset = MultiCondDataset(
        mri_dir="/home/hekim/train/mri",
        pet_dir="/home/hekim/train/pet",
        roi_map_path=config.roi_map_path,
        caption_path="/home/hekim/caption_train_with_labels.txt",
        tokenizer=tokenizer,
        size=(config.image_size, config.image_size),
        train_mode=True,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    val_dataset = MultiCondDataset(
        mri_dir="/home/hekim/valid/mri",
        pet_dir="/home/hekim/valid/pet",
        roi_map_path=config.roi_map_path,
        caption_path="/home/hekim/caption_valid_with_labels.txt",
        tokenizer=tokenizer,
        size=(config.image_size, config.image_size),
        train_mode=False,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, drop_last=True
    )

    output_dir = f"./outputs/{config.residual_mode}_lam{config.lambda_roi_loss}"
    os.makedirs(output_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")
    best_ckpt_path = os.path.join(output_dir, "best_checkpoint.pt")
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(latest_ckpt_path):
        print(f"Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path)
        unet.load_state_dict(checkpoint['unet_state_dict'])
        if 'text_encoder_state_dict' in checkpoint and config.clip_finetune_strategy != "none":
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from Epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, config.epochs):
        unet.train()
        if config.clip_finetune_strategy != "none":
            text_encoder.train()

        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{config.epochs} [Train]")
        total_train_loss = 0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            source_mri = batch["mri"].to(device)
            target_images = batch["pet"].to(device)
            roi_map = batch["roi_map"].to(device)
            batch_size = source_mri.shape[0]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if config.clip_finetune_strategy != "none":
                text_context = text_encoder(input_ids, attention_mask=attention_mask)[0]
            else:
                with torch.no_grad():
                    text_context = text_encoder(input_ids, attention_mask=attention_mask)[0]

            # Classifier-free guidance: drop the textual condition with 10% probability.
            if random.random() < 0.1:
                uncond_input = tokenizer(
                    [""] * batch_size, padding="max_length",
                    max_length=tokenizer.model_max_length, return_tensors="pt",
                )
                with torch.no_grad():
                    context = text_encoder(
                        uncond_input.input_ids.to(device),
                        attention_mask=uncond_input.attention_mask.to(device),
                    )[0]
            else:
                context = text_context

            # Forward process: pick a random t, then build noisy_images = PET + eta_t * RD + scaled noise.
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()

            residual_e0 = compute_residual(source_mri, target_images, roi_map=roi_map, eps=1e-8)

            if step % 50 == 0:
                save_debug_visualization(source_mri, target_images, roi_map, residual_e0, output_dir, epoch, step)

            eta_t = (timesteps / (noise_scheduler.config.num_train_timesteps - 1)).float().view(-1, 1, 1, 1)
            interpolated_images = target_images + eta_t * residual_e0

            noise_level = (1 - noise_scheduler.alphas_cumprod.to(device)).sqrt()[timesteps].view(-1, 1, 1, 1)
            noise = torch.randn_like(target_images)
            noisy_images = interpolated_images + noise * noise_level * config.noise_strength

            predicted_images_x0 = unet(noisy_images, timesteps, encoder_hidden_states=context).sample

            loss = pwr_loss(
                predicted_images_x0, target_images, roi_map,
                lambda_roi=config.lambda_roi_loss,
                use_roi_weight=config.use_roi_weighted_loss,
            )
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix(train_loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        progress_bar.close()

        # --- Validation ---
        unet.eval()
        if config.clip_finetune_strategy != "none":
            text_encoder.eval()

        total_val_loss = 0
        val_progress_bar = tqdm(total=len(val_dataloader), desc=f"Epoch {epoch + 1}/{config.epochs} [Valid]")

        with torch.no_grad():
            for batch in val_dataloader:
                source_mri = batch["mri"].to(device)
                target_images = batch["pet"].to(device)
                roi_map = batch["roi_map"].to(device)
                batch_size_val = source_mri.shape[0]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                context = text_encoder(input_ids, attention_mask=attention_mask)[0]
                # Fixed timestep (500) for consistent epoch-to-epoch comparison.
                timesteps = torch.full((batch_size_val,), 500, device=device).long()

                residual_e0 = compute_residual(source_mri, target_images, roi_map=roi_map, eps=1e-8)
                eta_t = (timesteps / (noise_scheduler.config.num_train_timesteps - 1)).float().view(-1, 1, 1, 1)
                interpolated_images = target_images + eta_t * residual_e0
                noise_level = (1 - noise_scheduler.alphas_cumprod.to(device)).sqrt()[timesteps].view(-1, 1, 1, 1)
                noise = torch.randn_like(target_images)
                noisy_images = interpolated_images + noise * noise_level * config.noise_strength
                predicted_images_x0 = unet(noisy_images, timesteps, encoder_hidden_states=context).sample

                current_val_loss = pwr_loss(
                    predicted_images_x0, target_images, roi_map,
                    lambda_roi=config.lambda_roi_loss,
                    use_roi_weight=config.use_roi_weighted_loss,
                )
                total_val_loss += current_val_loss.item()
                val_progress_bar.update(1)
                val_progress_bar.set_postfix(val_loss=current_val_loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_progress_bar.close()
        print(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        # --- Save checkpoints ---
        current_checkpoint = {
            'epoch': epoch + 1,
            'unet_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config,
        }
        if config.clip_finetune_strategy != "none":
            current_checkpoint['text_encoder_state_dict'] = text_encoder.state_dict()
        torch.save(current_checkpoint, latest_ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            current_checkpoint['best_val_loss'] = best_val_loss
            torch.save(current_checkpoint, best_ckpt_path)
            print(f"New best model. Saved to {best_ckpt_path}")

    # --- Final inference-ready model ---
    print("\nTraining finished. Saving final model for inference.")
    final_model_dir = os.path.join(output_dir, "final_model_for_inference")
    os.makedirs(final_model_dir, exist_ok=True)

    final_ckpt_path = best_ckpt_path if os.path.exists(best_ckpt_path) else latest_ckpt_path
    if os.path.exists(final_ckpt_path):
        checkpoint = torch.load(final_ckpt_path)
        unet.load_state_dict(checkpoint['unet_state_dict'])
        if 'text_encoder_state_dict' in checkpoint:
            text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])

    unet.save_pretrained(os.path.join(final_model_dir, "unet"))
    text_encoder.save_pretrained(os.path.join(final_model_dir, "text_encoder"))
    tokenizer.save_pretrained(os.path.join(final_model_dir, "tokenizer"))
    print(f"Final inference-ready model saved to {final_model_dir}")
