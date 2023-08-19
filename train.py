import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file
from torch.utils.data import DataLoader, Dataset

from upscaler import LatentUpscaler as Upscaler
from vae import get_vae

torch.backends.cudnn.benchmark = True

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("--steps", type=int, default=500000, help="No. of training steps")
	parser.add_argument('--bs', type=int, default=4, help="Batch size")
	parser.add_argument('--lr', default="5e-4", help="Learning rate")
	parser.add_argument("-n", "--save_every_n", type=int, dest="save", default=50000, help="Save model/sample periodically")
	parser.add_argument("-r", "--res", type=int, default=512, help="Source resolution")
	parser.add_argument("-f", "--fac", type=float, default=1.5, help="Upscale factor")
	parser.add_argument("-v", "--ver", choices=["v1","xl"], default="v1", help="SD version")
	parser.add_argument('--vae', help="Path to VAE (Optional)")
	parser.add_argument('--resume', help="Checkpoint to resume from")
	args = parser.parse_args()
	try:
		float(args.lr)
	except:
		parser.error("--lr must be a valid float eg. 0.001 or 1e-3")
	return args

vae = None
def sample_decode(latent, filename, version):
	global vae
	if not vae:
		vae = get_vae(version, fp16=True)
		vae.to("cuda")

	latent = latent.half().to("cuda")
	out = vae.decode(latent).sample
	out = out.cpu().detach().numpy()
	out = np.squeeze(out, 0)
	out = out.transpose((1, 2, 0))
	out = np.clip(out, -1.0, 1.0)
	out = (out+1)/2 * 255
	out = out.astype(np.uint8)
	out = Image.fromarray(out)
	out.save(filename)

def eval_model(step, model, criterion, scheduler, src, dst):
	with torch.no_grad():
		t_pred = model(src)
		t_loss = criterion(t_pred, dst)
	tqdm.write(f"{str(step):<10} {loss.data.item():.4e}|{t_loss.data.item():.4e} @ {float(scheduler.get_last_lr()[0]):.4e}")
	log.write(f"{step},{loss.data.item()},{t_loss.data.item()},{float(scheduler.get_last_lr()[0])}\n")
	log.flush()

def save_model(step, model, ver, fac, src):
	out = model(src)
	output_name = f"./models/latent-upscaler_SD{ver}-x{fac}_e{round(step/1000)}k"
	sample_decode(out, f"{output_name}.png", ver)
	save_file(model.state_dict(), f"{output_name}.safetensors")

class Latent:
	def __init__(self, md5, ver, src_res, dst_res):
		src = os.path.join(f"latents/{ver}_{src_res}px", f"{md5}.npy")
		dst = os.path.join(f"latents/{ver}_{dst_res}px", f"{md5}.npy")
		self.src = torch.from_numpy(np.load(src)).to("cuda")
		self.dst = torch.from_numpy(np.load(dst)).to("cuda")
		self.src = torch.squeeze(self.src, 0)
		self.dst = torch.squeeze(self.dst, 0)

class LatentDataset(Dataset):
	def __init__(self, ver, src_res, dst_res):
		print("Loading latents from disk")
		self.latents = []
		for i in tqdm(os.listdir(f"latents/{ver}_{src_res}px")):
			md5 = os.path.splitext(i)[0]
			self.latents.append(
				Latent(md5, ver, src_res, dst_res)
			)

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		return (
			self.latents[index].src,
			self.latents[index].dst,
		)

if __name__ == "__main__":
	args = parse_args()
	target_dev = "cuda"
	dst_res = int(args.res*args.fac)

	dataset = LatentDataset(args.ver, args.res, dst_res)
	loader = DataLoader(
		dataset,
		batch_size=args.bs,
		shuffle=True,
		num_workers=0,
	)	

	if not os.path.isdir("models"): os.mkdir("models")
	log = open(f"models/latent-upscaler_SD{args.ver}-x{args.fac}.csv", "w")

	if os.path.isfile(f"test_{args.ver}_{args.res}px.npy") and os.path.isfile(f"test_{args.ver}_{dst_res}px.npy"):
		eval_src = torch.from_numpy(np.load(f"test_{args.ver}_{args.res}px.npy")).to(target_dev)
		eval_dst = torch.from_numpy(np.load(f"test_{args.ver}_{dst_res}px.npy")).to(target_dev)
	else:
		eval_src = torch.unsqueeze(dataset[0][0],0)
		eval_dst = torch.unsqueeze(dataset[0][1],0)

	model = Upscaler(args.fac)
	if args.resume:
		model.load_state_dict(load_file(args.resume))
	model.to(target_dev)

	# criterion = torch.nn.MSELoss()
	criterion = torch.nn.L1Loss()

	# optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr)/args.bs)
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr)/args.bs)

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
		optimizer,
		total_steps=int(args.steps/args.bs),
		max_lr=float(args.lr)/args.bs,
		pct_start=0.015,
		final_div_factor=2500,
	)
	# scaler = torch.cuda.amp.GradScaler()
	progress = tqdm(total=args.steps)

	while progress.n < args.steps:
		for src, dst in loader:
			with torch.cuda.amp.autocast():
				y_pred = model(src) # forward
				loss = criterion(y_pred, dst) # loss

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			# eval/save
			progress.update(args.bs)
			if progress.n % (1000 + 1000%args.bs) == 0:
				eval_model(progress.n, model, criterion, scheduler, eval_src, eval_dst)
			if progress.n % (args.save + args.save%args.bs) == 0:
				save_model(progress.n, model, args.ver, args.fac, eval_src)
			if progress.n >= args.steps:
				break
	progress.close()

	# save final output
	eval_model(args.steps, model, criterion, scheduler, eval_src, eval_dst)
	save_model(args.steps, model, args.ver, args.fac, eval_src)
	log.close()
