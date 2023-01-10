import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import importlib
import models
importlib.reload(models)
from models import DiTLN

# data
dataset = MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.tile(x,(4,1,1)))]))
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=64)
val_loader = DataLoader(mnist_val, batch_size=64)

# next(iter(train_loader))[0].shape

# model
# image_size = 256
image_size = 28
# assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
# latent_size = image_size // 8
latent_size = image_size
model = DiTLN(latent_size=latent_size)

# training
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
