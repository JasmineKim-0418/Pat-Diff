# PaT-Diff: Pathology-Aware Residual Diffusion Framework for MRI-to-PET Translation in Alzheimer's Disease

##
This is the official code repository for our paper: **PaT-Diff: Pathology-Aware Residual Diffusion Framework for MRI-to-PET Translation in Alzheimer's Disease**.

## Motivation
PET and MRI provide complementary information for Alzheimer's disease (AD) diagnosis — PET visualizes molecular abnormalities such as β-amyloid accumulation, while MRI captures detailed anatomical structure. However, acquiring paired MRI–PET scans is often impractical due to PET's high cost and radiation exposure. Most prior MRI-to-PET translation methods rely on GAN-based models, which often suffer from structural inconsistencies or fail to reproduce AD-specific pathological patterns.

We propose **PaT-Diff**, a pathology-aware residual diffusion framework that learns AD-related residual representations between the anatomical MRI and pathological PET domains. PaT-Diff combines (1) a dual-domain residual that mixes pixel-domain and log-domain residuals through an AD-effect map, (2) a CLIP-conditioned clinical guidance module (CLIP-CGM) that integrates demographic attributes, and (3) a Pathology-Weighted Residual (PWR) loss that emphasizes amyloid-relevant regions. Experiments on ADNI demonstrate that PaT-Diff outperforms GAN- and diffusion-based baselines in both image fidelity and pathological alignment.

## 📁 Repository Structure
```
PaT-Diff/                    # Root directory
├── README.md
├── train.py                 # Training entry-point
│                            #  - MultiCondDataset (paired MRI/PET + ROI map + caption)
│                            #  - compute_residual (log_rd_on_roi: hybrid pixel/log residual)
│                            #  - PWR loss (ROI-weighted MSE + MAE)
│                            #  - Stable-Diffusion U-Net + CLIP text encoder + AutoencoderKL
├── inference.py             # Inference entry-point
│                            #  - DDIM sampling with classifier-free guidance
│                            #  - log-space → pixel-space recovery (exp − 1)
│                            #  - PSNR / SSIM / MSE per diagnostic group
└── requirements.txt         # Required Python packages
```

## Environment Setup
- Python 3.9+
- CUDA 11.7+
- PyTorch 2.0+
- 🤗 diffusers, transformers
- Required libraries are listed in `requirements.txt`.

```
conda create -n patdiff python=3.9
conda activate patdiff
pip install -r requirements.txt
```

## Data Preparation
- Download paired MRI–PET scans from the **ADNI** database ([ida.loni.usc.edu](https://ida.loni.usc.edu/)).
- All volumes are spatially normalized to the **MNI152** template via affine registration.
- MRI volumes are scaled to [0, 1] and then rescaled to [−1, 1]; PET images are directly normalized to [−1, 1].
- Data split: 823 training / 101 validation / 104 testing pairs.

Expected directory layout:
```
<data_root>/
├── train/{mri,pet}/   # *.npy   (256×256 axial slices)
├── val/{mri,pet}/
├── test/{mri,pet}/
└── captions/
    ├── caption_train.txt   # "<mri_filename>|<clinical_caption>|<label>"   per line
    ├── caption_val.txt
    └── caption_test.txt
```

You will also need the **AD-effect map** (`.nii.gz`) from Oh et al. (NeuroImage 2025) — set the path via `Config.roi_map_path` in `train.py`.

## Training
Edit the `Config` block at the top of `train.py` to point to your data and AD-effect map, then run:
```
python train.py
```
Checkpoints and the final fused model are saved under `./outputs/log_rd_on_roi_lam{λ_ROI}/`.

Default hyperparameters follow the paper: Adam (lr = 1×10⁻⁵), batch size 8, 150 epochs, log clipping τ = 3, λ_MAE = 0.5, λ_ROI = 0.2, classifier-free guidance dropout 10%.

## Inference
```
python inference.py \
    --model_dir ./outputs/log_rd_on_roi_lam0.2/final_model_for_inference \
    --data_dir /path/to/test \
    --caption_path /path/to/caption_test.txt \
    --output_dir ./inference_results \
    --guidance_scale 2.0 \
    --num_inference_steps 50
```
The script saves a 4-panel comparison plot per slice (MRI · GT PET · synthesized PET · residual) and prints PSNR / SSIM / MSE per diagnostic group (CN / SMC / EMCI / MCI / LMCI / AD).

## Acknowledgements

We acknowledge the use of the following publicly available dataset:
- [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://ida.loni.usc.edu/) — paired MRI–PET scans

This codebase builds on:
- [Stable Diffusion v1.4](https://github.com/CompVis/stable-diffusion) / [🤗 diffusers](https://github.com/huggingface/diffusers) — U-Net backbone and AutoencoderKL
- [OpenAI CLIP](https://github.com/openai/CLIP) — text encoder for CLIP-CGM
- AD-effect map from [Oh et al., *NeuroImage* 2025](https://doi.org/10.1016/j.neuroimage.2025.121077) — *A quantitatively interpretable model for Alzheimer's disease prediction using deep counterfactuals*

This work is also inspired by prior MRI-to-PET translation studies:
- [Pix2Pix](https://github.com/phillipi/pix2pix) [CVPR 2017]
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) [ICCV 2017]
- [BBDM: Brownian Bridge Diffusion Model](https://github.com/xuekt98/BBDM) [CVPR 2023]

## Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026patdiff,
  author    = {Kim, Ha-Eun and Oh, Kwanseok and Suk, Heung-Il},
  title     = {PaT-Diff: Pathology-Aware Residual Diffusion Framework for MRI-to-PET Translation in Alzheimer's Disease},
  year      = {2026},
}
```
