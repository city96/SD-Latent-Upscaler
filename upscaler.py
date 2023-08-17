import torch
import torch.nn as nn
import numpy as np

class LatentUpscaler(nn.Module):
	def head(self):
		return [
			nn.Conv2d(self.chan, self.size, kernel_size=self.krn, padding=self.pad),
			nn.ReLU(),
			nn.Upsample(scale_factor=self.fac, mode="nearest"),
			nn.ReLU(),
		]
	def core(self):
		layers = []
		for _ in range(self.depth):
			layers += [
				nn.Conv2d(self.size, self.size, kernel_size=self.krn, padding=self.pad),
				nn.ReLU(),
			]
		return layers
	def tail(self):
		return [
			nn.Conv2d(self.size, self.chan, kernel_size=self.krn, padding=self.pad),
		]

	def __init__(self, fac, depth=16):
		super().__init__()
		self.size = 64      # Conv2d size
		self.chan = 4       # in/out channels
		self.depth = depth  # no. of layers
		self.fac = fac      # scale factor
		self.krn = 3        # kernel size
		self.pad = 1        # padding

		self.sequential = nn.Sequential(
			*self.head(),
			*self.core(),
			*self.tail(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.sequential(x)
