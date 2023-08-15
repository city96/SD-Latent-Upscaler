import os
import torch
import hashlib
import argparse
import numpy as np
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from PIL import Image

from vae import get_vae

def parse_args():
	parser = argparse.ArgumentParser(description="Preprocess images into latents")
	parser.add_argument("-r", "--res", type=int, default=512, help="Source resolution")
	parser.add_argument("-f", "--fac", type=float, default=1.5, help="Upscale factor")
	parser.add_argument("-v", "--ver", choices=["v1","xl"], default="v1", help="SD version")
	parser.add_argument('--vae', help="Path to VAE (Optional)")
	parser.add_argument('--src', default="raw", help="Source folder with images")
	return parser.parse_args()

def encode(vae, img):
	"""image [PIL Image] -> latent [np array]"""
	inp = transforms.ToTensor()(img).unsqueeze(0)
	inp = inp.to("cuda") # move to GPU
	latent = vae.encode(inp*2.0-1.0)
	latent = latent.latent_dist.sample()
	return latent.cpu().detach()

def scale(path, res):
	"""Crop image to the top-left corner"""
	img = Image.open(path)
	img = img.convert('RGB')
	target = (res, res)
	if min(img.height, img.width) < 256:
		return
	if img.width > img.height:
		target = (int(img.width/img.height*res), res)
	elif img.height > img.width:
		target = (res, int(img.height/img.width*res))
	img = img.resize(target, Image.LANCZOS)
	img = img.crop([0,0,res,res])
	return img

def process_folder(vae, src_dir, ver, res):
	dst_dir = f"latents/{ver}_{res}px"
	if not os.path.isdir(dst_dir):
		os.mkdir(dst_dir)

	for file in tqdm(os.listdir(src_dir)):
		src = os.path.join(src_dir, file)
		md5 = hashlib.md5(open(src,'rb').read()).hexdigest()
		dst = os.path.join(dst_dir, f"{md5}.npy")
		if os.path.isfile(dst):
			continue
		img = scale(src, res)
		latent = encode(vae, img)
		np.save(dst, latent)

def process_res(vae, src_dir, ver, res):
	process_folder(vae, src_dir, ver, res)
	# test image, optional
	if os.path.isfile("test.png"):
		if os.path.isfile(f"test_{ver}_{res}px.npy"):
			return
		img = scale("test.png", res)
		latent = encode(vae, img)
		np.save(f"test_{ver}_{res}px.npy", latent)
	torch.cuda.empty_cache()

if __name__ == "__main__":
	if not os.path.isdir("latents"):
		os.mkdir("latents")
	args = parse_args()
	vae = get_vae(args.ver, args.vae)
	vae.to("cuda")
	## args
	dst_res = int(args.res*args.fac)
	process_res(vae, args.src, args.ver, args.res)
	process_res(vae, args.src, args.ver, dst_res)
