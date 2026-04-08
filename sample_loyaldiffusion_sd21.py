import argparse
import json
import os
import sys

import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Repli_DiT"))

from models.rau_unet import RAUNet
from models.dual_unet import DualUNet


def load_unet(checkpoint_path, unet_type="standard", sc_block_indices=None):
    unet_dir = os.path.join(checkpoint_path, "unet")
    with open(os.path.join(unet_dir, "config.json")) as f:
        unet_config = json.load(f)
    clean = {k: v for k, v in unet_config.items() if not k.startswith("_")}
    weights = os.path.join(unet_dir, "diffusion_pytorch_model.bin")

    if unet_type == "raunet":
        unet = RAUNet(sc_block_indices=sc_block_indices or [2, 3], **clean)
        unet.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    else:
        unet = UNet2DConditionModel(**clean)
        unet.load_state_dict(torch.load(weights, map_location="cpu"))
    return unet


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.join(args.output_dir, "generations"), exist_ok=True)

    unet_std = load_unet(args.checkpoint_front, unet_type="standard")
    unet_rau = load_unet(args.checkpoint_tail, unet_type="raunet",
                         sc_block_indices=args.sc_block_indices)

    dual_unet = DualUNet.from_two_unets(unet_std, unet_rau, args.t_threshold)

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        unet=dual_unet, safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_front, subfolder="tokenizer", use_fast=False)

    if args.prompt is not None:
        prompt_list = [args.prompt] * args.num_images
    elif args.prompt_json is not None:
        with open(args.prompt_json) as f:
            all_prompts = json.load(f)
        prompts = [v[0] for v in all_prompts.values()]
        np.random.seed(args.seed)
        prompt_list = list(np.random.choice(prompts, args.num_images))
    else:
        prompt_list = ["An image"] * args.num_images

    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.writelines(p + "\n" for p in prompt_list)

    count = 0
    for i, prompt in enumerate(prompt_list):
        images = pipe(
            prompt=prompt,
            height=args.resolution, width=args.resolution,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.images_per_prompt,
        ).images
        for img in images:
            if img.size[0] > args.resolution:
                img = img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
            img.save(os.path.join(args.output_dir, "generations", f"{count:05d}.png"))
            count += 1
        if (i + 1) % 10 == 0:
            print(f"Generated {count} images")

    print(f"Done. {count} images saved to {args.output_dir}/generations/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_front", type=str, required=True)
    parser.add_argument("--checkpoint_tail", type=str, required=True)
    parser.add_argument("--t_threshold", type=int, default=500)
    parser.add_argument("--sc_block_indices", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_json", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./generated_images")
    args = parser.parse_args()
    main(args)
