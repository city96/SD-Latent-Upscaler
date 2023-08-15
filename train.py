import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from PIL import Image
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from upscaler import LatentUpscaler as Upscaler
from vae import get_vae

def parse_args():
	parser = argparse.ArgumentParser(description="Train latent interposer model")
	parser.add_argument("--steps", type=int, default=500000, help="No. of training steps")
	parser.add_argument('--bs', type=int, default=1, help="Batch size")
	parser.add_argument('--lr', default="1e-8", help="Learning rate")
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

class Latent:
	def __init__(self, md5, ver, src_res, dst_res, dev):
		src = os.path.join(f"latents/{ver}_{src_res}px", f"{md5}.npy")
		dst = os.path.join(f"latents/{ver}_{dst_res}px", f"{md5}.npy")
		self.src = torch.from_numpy(np.load(src)).to(dev)
		self.dst = torch.from_numpy(np.load(dst)).to(dev)

def load_latents(ver, src_res, dst_res, dev):
	print("Loading latents from disk")
	latents = []
	for i in tqdm(os.listdir(f"latents/{ver}_{src_res}px")):
		md5 = os.path.splitext(i)[0]
		latents.append(Latent(md5, ver, src_res, dst_res, dev))
	return latents

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

if __name__ == "__main__":
	args = parse_args()
	target_dev = "cuda"
	dst_res = int(args.res*args.fac)

	latents = load_latents(args.ver, args.res, dst_res, target_dev)

	if not os.path.isdir("models"): os.mkdir("models")
	log = open(f"models/latent-upscaler_SD{args.ver}-x{args.fac}.csv", "w")

	if os.path.isfile(f"test_{args.ver}_{args.res}px.npy") and os.path.isfile(f"test_{args.ver}_{dst_res}px.npy"):
		ss_latent = torch.from_numpy(np.load(f"test_{args.ver}_{args.res}px.npy")).to(target_dev)
		st_latent = torch.from_numpy(np.load(f"test_{args.ver}_{dst_res}px.npy")).to(target_dev)
	else:
		sample_latent = random.choice(latents)
		ss_latent = sample_latent.src.to(target_dev)
		st_latent = sample_latent.dst.to(target_dev)

	model = Upscaler(args.fac)
	if args.resume:
		model.load_state_dict(load_file(args.resume))
	model.to(target_dev)

	criterion = torch.nn.MSELoss(size_average=False)
	optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr)/args.bs)

	for t in tqdm(range(int(args.steps/args.bs)), unit_scale=args.bs):
		step = t*args.bs
		# input batch
		lts = [random.choice(latents) for _ in range(args.bs)]
		src = torch.cat([x.src for x in lts],0)
		dst = torch.cat([x.dst for x in lts],0)

		y_pred = model(src) # forward
		loss = criterion(y_pred, dst) # loss

		# backward
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# print loss
		if step%1000 == 0:
			# test loss
			with torch.no_grad():
				t_pred = model(ss_latent)
				t_loss = criterion(t_pred, st_latent)
			tqdm.write(f"{step} - {loss.data.item()/args.bs:.2f}|{t_loss.data.item()/args.bs:.2f}")
			log.write(f"{step},{loss.data.item()/args.bs:.2f},{t_loss.data.item()/args.bs:.2f}\n")
			log.flush()

		# sample/save
		if step%args.save == 0:
			out = model(ss_latent)
			output_name = f"./models/latent-upscaler_SD{args.ver}-x{args.fac}_e{step/1000}k"
			sample_decode(out, f"{output_name}.png", args.ver)
			save_file(model.state_dict(), f"{output_name}.safetensors")
	# save final output
	output_name = f"./models/latent-upscaler_SD{args.ver}-x{args.fac}_e{step/1000}k"
	sample_decode(out, f"{output_name}.png", args.ver)
	save_file(model.state_dict(), f"{output_name}.safetensors")
	log.close()
