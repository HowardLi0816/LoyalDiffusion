# LoyalDiffusion

Code for the paper **"Rethinking Skip Connections in Diffusion Models for Replication Mitigation"**.

> **Note:** The original source code was lost due to a server failure. This repository was reconstructed by [Claude Code](https://claude.ai/code) based on the paper and surviving code fragments from the research directory (`Repli_DiT/`). The core training logic and model architecture in `Repli_DiT/` are written by the paper authors. The top-level scripts (`train_loyaldiffusion_sd21.py`, `sample_loyaldiffusion_sd21.py`) and the `models/` module are the reconstructed clean interfaces.

---

## Method

LoyalDiffusion reduces data replication in fine-tuned diffusion models through two components:

**RAU-Net (Replication-Aware U-Net):** Inserts Conv 3×3 blocks on skip connections SC3 and SC4 of the SD2.1 U-Net encoder. This prevents low-level memorized features from being directly copied to the decoder.

**Two-stage timestep training:** Rather than training one model over all timesteps:
- A standard U-Net is fine-tuned on `t ∈ [0, τ)` (low-noise, detail refinement)
- A RAU-Net is fine-tuned on `t ∈ [τ, 1000)` (high-noise, global structure)

At inference a `DualUNet` wrapper routes each step to the appropriate model.

The method can also be combined with caption-based data augmentation (Generalized Captions / Dual Fusion).

---

## Repository Structure

```
LoyalDiffusion/
├── train_loyaldiffusion_sd21.py     # Training entry point
├── sample_loyaldiffusion_sd21.py    # Inference script
├── models/
│   ├── rau_unet.py                  # RAU-Net
│   └── dual_unet.py                 # Timestep router (DualUNet)
├── configs/
│   ├── sd21_baseline.yaml
│   ├── loyaldiffusion_stage1_front.yaml
│   └── loyaldiffusion_stage2_tail.yaml
└── Repli_DiT/                       # Original research code by the paper authors
    ├── diff_split_train.py          # Original training script
    ├── diff_split_inference.py      # Original inference script
    ├── customize_unet.py            # All UNet variants (RAU-Net, DualUNet, etc.)
    ├── datasets.py                  # Dataset utilities
    ├── diff_retrieval.py            # SSCD-based retrieval evaluation
    ├── cal_gen_score.py
    └── unet_config/unet_config.json
```

---

## Requirements

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install webdataset
pip install clip-anytorch
```

---

## Training

### Stage 1 — Standard UNet on low timesteps

```bash
accelerate launch train_loyaldiffusion_sd21.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
    --instance_data_dir ./laion_10k_data_2 \
    --output_dir ./checkpoints/stage1 \
    --training_phase front \
    --t_threshold 500 \
    --max_train_steps 100000 \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 5000 \
    --class_prompt laion_orig \
    --mixed_precision fp16
```

### Stage 2 — RAU-Net on high timesteps

```bash
accelerate launch train_loyaldiffusion_sd21.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
    --instance_data_dir ./laion_10k_data_2 \
    --output_dir ./checkpoints/stage2 \
    --training_phase tail \
    --t_threshold 500 \
    --modify_unet_tail \
    --partial_conv_block_list_tail 2 3 \
    --all_conv_type_tail single_conv \
    --max_train_steps 100000 \
    --train_batch_size 1 \
    --learning_rate 5e-6 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 5000 \
    --class_prompt laion_orig \
    --mixed_precision fp16
```

`--t_threshold` sets τ. Both stages must use the same value.

---

## Inference

```bash
python sample_loyaldiffusion_sd21.py \
    --checkpoint_front ./checkpoints/stage1/checkpoint_front \
    --checkpoint_tail  ./checkpoints/stage2/checkpoint_tail \
    --t_threshold 500 \
    --prompt_json ./laion_10k_data_2/laion_10k_data_2_combined_captions.json \
    --num_images 8201 \
    --output_dir ./generated_images \
    --resolution 512
```

---

## Evaluation

```bash
# Replication rate (SSCD nearest-neighbor retrieval)
python Repli_DiT/diff_retrieval.py \
    --arch resnet50_disc \
    --similarity_metric dotproduct \
    --pt_style sscd \
    --query_dir ./generated_images/generations \
    --val_dir ./laion_10k_data_2/

# FID
python -m pytorch_fid ./laion_10k_data_2/raw_images ./generated_images/generations --device cuda
```

---

## Data-side Enhancements

To combine with Generalized Captions or Dual Fusion augmentation:

```bash
# Dual Fusion
--dual_fusion --cat_object_latent --tiny_path ./tiny-imagenet-200/

# Generalized Captions
--use_clean_prompts
```

## RepliBing Dataset is Released in the Google Drive

https://drive.google.com/drive/folders/1fbXncTKn4cfGjaOomxK-XuWN21_SgtI5?usp=sharing
