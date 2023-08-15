import torch
import torch.nn as nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


class Upscaler(nn.Module):
	"""
		Basic NN layout, ported from:
		https://github.com/city96/SD-Latent-Upscaler/blob/main/upscaler.py
	"""
	version = 1.0 # network revision
	def __init__(self, fac):
		super().__init__()

		module_list = [
			nn.Conv2d(4, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Upsample(scale_factor=fac, mode="nearest"),
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


class LatentUpscaler:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(s):
		return {
			"required": {
				"samples": ("LATENT", ),
				"latent_ver": (["v1", "xl"],),
				"scale_factor": (["1.25", "1.5", "2.0"],),
			}
		}

	RETURN_TYPES = ("LATENT",)
	FUNCTION = "upscale"
	CATEGORY = "latent"

	def upscale(self, samples, latent_ver, scale_factor):
		model = Upscaler(scale_factor)
		weights = str(hf_hub_download(
			repo_id="city96/SD-Latent-Upscaler",
			filename=f"latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors")
		)
		# weights = f"./latent-upscaler-v{model.version}_SD{latent_ver}-x{scale_factor}.safetensors"

		model.load_state_dict(load_file(weights))
		lt = samples["samples"]
		lt = model(lt)
		del model
		return ({"samples": lt},)

NODE_CLASS_MAPPINGS = {
	"LatentUpscaler": LatentUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentUpscaler": "Latent Upscaler"
}
