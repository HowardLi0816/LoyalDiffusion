import argparse
import itertools
import math
import os
import json
import random
import numpy as np

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Repli_DiT"))

from models.rau_unet import RAUNet
from models.dual_unet import DualUNet

try:
    from Repli_DiT.datasets import ObjectAttributeDataset, get_classnames, Webdatasetloader
except ImportError:
    from datasets import ObjectAttributeDataset, get_classnames, Webdatasetloader

logger = get_logger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def import_text_encoder_cls(pretrained_model_name_or_path, revision):
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    model_class = config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    raise ValueError(f"Unsupported text encoder: {model_class}")


def calss_name_dict(path):
    class_dict = {}
    with open(os.path.join(path, "words.txt")) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                class_dict[parts[0]] = parts[1].split(", ")
    return class_dict


def get_rand_tinyI(tiny_path, class_dict, batch_size, size, tokenizer, device,
                   center_crop=False, random_flip=False):
    path = os.path.join(tiny_path, "train")
    ids_list, img_list = [], []
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    for _ in range(batch_size):
        d = random.choice([x for x in os.scandir(path) if x.is_dir()]).name
        f = random.choice(list(os.scandir(os.path.join(path, d, "images")))).name
        name = random.choice(class_dict[d])
        cap = tokenizer(" with " + name, truncation=True, padding="max_length",
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
        ids_list.append(cap)
        img = Image.open(os.path.join(path, d, "images", f)).convert("RGB")
        img_list.append(tf(img))
    imgs = torch.stack(img_list).to(memory_format=torch.contiguous_format).float().to(device)
    tokens = torch.cat(ids_list).to(device)
    return imgs, tokens


def add_tensor_with_weight(t1, t2, w1):
    return t1 * w1 + t2 * (1 - w1)


def collate_fn(examples):
    input_ids = torch.cat([e["instance_prompt_ids"] for e in examples])
    pixel_values = torch.stack([e["instance_images"] for e in examples])
    return {"input_ids": input_ids, "pixel_values": pixel_values.to(memory_format=torch.contiguous_format).float()}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_prompt_loc", type=str, default=None)
    parser.add_argument("--class_prompt", type=str, default="nolevel",
                        choices=["nolevel", "classlevel", "instancelevel_blip",
                                 "instancelevel_random", "laion_orig"])
    parser.add_argument("--output_dir", type=str, default="loyal-diffusion-output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation_seed", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=10000)
    parser.add_argument("--modelsavesteps", type=int, default=20000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=5000)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("-j", "--num_workers", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--duplication", type=str, default="nodup",
                        choices=["nodup", "dup_both", "dup_image"])
    parser.add_argument("--weight_pc", type=float, default=0.05)
    parser.add_argument("--dup_weight", type=int, default=5)
    parser.add_argument("--rand_noise_lam", type=float, default=0)
    parser.add_argument("--mixup_noise_lam", type=float, default=0)
    parser.add_argument("--trainspecial", type=str, default=None,
                        choices=["allcaps", "randrepl", "randwordadd", "wordrepeat"])
    parser.add_argument("--trainspecial_prob", type=float, default=0.1)
    parser.add_argument("--trainsubset", type=float, default=None)
    parser.add_argument("--use_clean_prompts", action="store_true")
    parser.add_argument("--multiple_bilp_caption", action="store_true")
    parser.add_argument("--dual_fusion", action="store_true")
    parser.add_argument("--cat_object_latent", action="store_true")
    parser.add_argument("--tiny_path", type=str, default="./tiny-imagenet-200/")
    parser.add_argument("--caption_add_method", type=str, default="no_add",
                        choices=["no_add", "tail_add", "embed_add"])
    parser.add_argument("--latent_tiny_weight", type=float, default=0.5)
    parser.add_argument("--caption_tiny_weight", type=float, default=0.5)

    # Two-stage training
    parser.add_argument("--t_threshold", type=int, default=500,
                        help="Timestep boundary tau. Stage front trains [0, tau), tail trains [tau, 1000).")
    parser.add_argument("--training_phase", type=str, default="front", choices=["front", "tail"])
    parser.add_argument("--unet_config_path", type=str,
                        default="./Repli_DiT/unet_config/unet_config.json")
    parser.add_argument("--unet_pretrain_path", type=str, default=None)

    # RAU-Net options (used when training_phase=tail)
    parser.add_argument("--modify_unet_tail", action="store_true")
    parser.add_argument("--partial_conv_block_list_tail", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--all_conv_type_tail", type=str, default="single_conv",
                        choices=["single_conv", "single_conv_silu", "skip_unet", "max_pooling", "custom_conv"])

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(f"{args.output_dir}/generations", exist_ok=True)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer",
            revision=args.revision, use_fast=False)

    text_encoder_cls = import_text_encoder_cls(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.training_phase == "front":
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
    else:
        if args.modify_unet_tail:
            with open(args.unet_config_path) as f:
                unet_config = json.load(f)
            clean = {k: v for k, v in unet_config.items() if not k.startswith("_")}
            unet = RAUNet(sc_block_indices=args.partial_conv_block_list_tail, **clean)
            if args.unet_pretrain_path:
                unet.load_sd21_weights(args.unet_pretrain_path)
            else:
                sd21 = UNet2DConditionModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)
                missing, _ = unet.load_state_dict(sd21.state_dict(), strict=False)
                print(f"[RAUNet] {len(missing)} keys initialized randomly (new conv layers)")
                del sd21
        else:
            unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate *= (
            args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    params = itertools.chain(unet.parameters(), text_encoder.parameters()) \
        if args.train_text_encoder else unet.parameters()
    optimizer = optimizer_cls(params, lr=args.learning_rate,
                              betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay,
                              eps=args.adam_epsilon)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    is_laion = "laion" in args.instance_data_dir
    if not is_laion:
        train_dataset = ObjectAttributeDataset(
            instance_data_root=args.instance_data_dir,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            prompt_json=args.instance_prompt_loc,
            duplication=args.duplication,
            args=args,
        )
        if args.duplication in ["dup_both", "dup_image"]:
            sampler = torch.utils.data.WeightedRandomSampler(
                train_dataset.samplingweights, len(train_dataset), replacement=True)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.train_batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=args.num_workers, sampler=sampler)
        else:
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.train_batch_size, shuffle=True,
                collate_fn=collate_fn, num_workers=args.num_workers)
    else:
        tars = []
        for root, _, files in os.walk(args.instance_data_dir):
            for f in files:
                if f.endswith(".tar"):
                    tars.append(os.path.join(root, f))
        train_dataloader = Webdatasetloader(
            tars, tokenizer=tokenizer, size=args.resolution,
            batch_size=args.train_batch_size, num_prepro_workers=args.num_workers,
            enable_text=True, enable_image=True, enable_metadata=False,
            wds_image_key="jpg", wds_caption_key="txt",
            center_crop=args.center_crop, random_flip=args.random_flip,
            cache_path=None, duplication=args.duplication,
            weight_pc=args.weight_pc, dup_weight=args.dup_weight, seed=args.seed,
            use_clean_prompts=args.use_clean_prompts,
            use_multiple_bilp_caption=args.multiple_bilp_caption)

    if args.max_train_steps is None:
        steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        try:
            accelerator.init_trackers("loyaldiffusion_sd21", config=vars(args))
        except Exception:
            pass

    logger.info(f"Training phase: {args.training_phase}, t_threshold: {args.t_threshold}")
    logger.info(f"Max steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        if args.dual_fusion and args.cat_object_latent:
            class_dict = calss_name_dict(args.tiny_path)

        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * 0.18215

                if args.dual_fusion and args.cat_object_latent:
                    bsz_ = latents.size(0)
                    tiny_imgs, class_tokens = get_rand_tinyI(
                        args.tiny_path, class_dict, bsz_, args.resolution,
                        tokenizer, latents.device, args.center_crop, args.random_flip)
                    if args.caption_add_method == "tail_add":
                        from Repli_DiT.diff_split_train import token_add_object
                        batch["input_ids"] = token_add_object(batch["input_ids"], class_tokens)
                    obj_latents = vae.encode(
                        tiny_imgs.to(dtype=weight_dtype)).latent_dist.sample() * 0.18215
                    latents = add_tensor_with_weight(latents, obj_latents, 1 - args.latent_tiny_weight)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                if args.training_phase == "front":
                    timesteps = torch.randint(0, args.t_threshold, (bsz,), device=latents.device).long()
                else:
                    timesteps = torch.randint(
                        args.t_threshold, noise_scheduler.config.num_train_timesteps,
                        (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                if args.dual_fusion and args.cat_object_latent and args.caption_add_method == "embed_add":
                    enc_class = text_encoder(class_tokens)[0]
                    encoder_hidden_states = add_tensor_with_weight(
                        encoder_hidden_states, enc_class, 1 - args.caption_tiny_weight)

                if args.rand_noise_lam > 0:
                    encoder_hidden_states += args.rand_noise_lam * torch.randn_like(encoder_hidden_states)
                if args.mixup_noise_lam > 0:
                    lam = np.random.beta(args.mixup_noise_lam, 1)
                    idx = torch.randperm(encoder_hidden_states.shape[0]).to(encoder_hidden_states.device)
                    encoder_hidden_states = lam * encoder_hidden_states + (1 - lam) * encoder_hidden_states[idx]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_clip = itertools.chain(unet.parameters(), text_encoder.parameters()) \
                        if args.train_text_encoder else unet.parameters()
                    accelerator.clip_grad_norm_(params_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.modelsavesteps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint_{args.training_phase}_{global_step}")
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        text_encoder=accelerator.unwrap_model(text_encoder),
                        revision=args.revision, safety_checker=None)
                    pipeline.save_pretrained(save_dir)

            progress_bar.set_postfix(loss=loss.detach().item(), lr=lr_scheduler.get_last_lr()[0])
            accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                            step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, f"checkpoint_{args.training_phase}")
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision, safety_checker=None)
        pipeline.save_pretrained(final_dir)
        print(f"Saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
