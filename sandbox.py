# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2


# Setup PyTorch:
torch.manual_seed(0)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device "+device)
num_sampling_steps = 250
cfg_scale = 4.0

torch.__version__

# Load model:
# image_size = 256
# assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
# latent_size = image_size // 8
# model = DiT_XL_2(input_size=latent_size).to(device)
# state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
# model.load_state_dict(state_dict)
from models import DiTLN

image_size = 28
# assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
# latent_size = image_size // 8
latent_size = image_size
model = DiTLN.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=0-step=860.ckpt", latent_size=latent_size)
model.eval()
model.to(device)
diffusion = create_diffusion(str(num_sampling_steps))
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)

# Labels to condition the model with:
# class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
class_labels = [0,1,2,3,4,5,6,7,8,9]

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
