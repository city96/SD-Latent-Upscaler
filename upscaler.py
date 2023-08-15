import torch
import torch.nn as nn
import numpy as np

class LatentUpscaler(nn.Module):
	def __init__(self, fac):
		super().__init__()

		module_list = [
			nn.Conv2d(4, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Upsample(scale_factor=fac, mode="nearest"), # bicubic was blurry
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(64, 32, kernel_size=7, padding=3),
			nn.ReLU(),
			nn.Conv2d(32, 4, kernel_size=5, padding=2),
		]
		self.sequential = nn.Sequential(*module_list)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.sequential(x)
