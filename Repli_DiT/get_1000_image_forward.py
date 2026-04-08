import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import os

# Set paths and load the model
model_path = "./out_laion_10k_baseline_orig_capiton_laion_orig_nodup/checkpoint/"  # Replace with your model path
output_dir = "./forward_images/2592/"
os.makedirs(output_dir, exist_ok=True)

# Load the pipeline with the specified scheduler
#scheduler = DDIMScheduler.from_config("./out_laion_10k_baseline_orig_capiton_laion_orig_nodup/checkpoint/scheduler/config.json")
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
pipe.to("cuda")  # Use GPU if available

# Load and preprocess the image
image_path = "./laion_10k_data_2/raw_images/000009664.jpg"
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Adjust size if necessary
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
image_tensor = preprocess(image).unsqueeze(0).to("cuda")

# Encode the image to latent space using VAE
with torch.no_grad():
    latents = pipe.vae.encode(image_tensor * 2 - 1).latent_dist.sample()

# Add noise over 1000 timesteps and decode each step
for t in range(1000):
    t = torch.tensor([t], device=latents.device)
    # Add noise to the latent according to the scheduler
    noise = torch.randn_like(latents)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

    # Decode the noisy latent back to image space
    with torch.no_grad():
        decoded_image = pipe.vae.decode(noisy_latents).sample
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)  # Rescale to [0, 1]

    # Convert to PIL image and save
    decoded_image_pil = transforms.ToPILImage()(decoded_image.squeeze().cpu())
    decoded_image_pil.save(os.path.join(output_dir, f"{int(t)+1:04d}.png"))

print(f"Images saved to {output_dir}")