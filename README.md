# SD-Latent-Upscaler
Upscaling stable diffusion latents using a small neural network.

Very similar to my [latent interposer](https://github.com/city96/SD-Latent-Interposer/tree/main), this small model can be used to upscale latents in a way that doesn't ruin the image. I mostly explain some of the issues with upscaling latents in [this issue](https://github.com/city96/SD-Advanced-Noise/issues/1#issuecomment-1678193121). Think of this as an ESRGAN for latents, except severely undertrained.

**Currently, SDXL has some minimal hue shift issues.** Because of course it does.

## Installation

### ComfyUI

To install it, simply clone this repo to your custom_nodes folder using the following command: `git clone https://github.com/city96/SD-Latent-Upscaler custom_nodes/SD-Latent-Upscaler`.

Alternatively, you can download the [comfy_latent_upscaler.py](https://github.com/city96/SD-Latent-Upscaler/blob/main/comfy_latent_upscaler.py) file to your ComfyUI/custom_nodes folder as well. You may need to install hfhub using the command pip install huggingface-hub inside your venv.

If you need the model weights for something else, they are [hosted on HF](https://huggingface.co/city96/SD-Latent-Upscaler/tree/main) under the same Apache2 license as the rest of the repo.

### Auto1111

Currently not supported but it should be possible to use it at the hires-fix part.

### Local models

The node pulls the required files from huggingface hub by default. You can create a `models` folder and place the modules there if you have a flaky connection or prefer to use it completely offline, it will load them locally instead. The path should be: `ComfyUI/custom_nodes/SD-Latent-Upscaler/models`

Alternatively, just clone the entire HF repo to it: `git clone https://huggingface.co/city96/SD-Latent-Upscaler custom_nodes/SD-Latent-Upscaler/models`

### Usage/principle

Usage is fairly simple. You use it anywhere where you would upscale a latent. If you need a higher scale factor (e.g. x4), simply chain two of the upscalers.

![LATENT_UPSCALER_ANI](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/dc187631-fd94-445e-9f20-a5741091bb0e)

![LATENT_UPSCALER_V2](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/16e7fcb3-74e5-476f-8d54-1eb4d6d4f78b)

As part of a workflow - notice how the second stage works despite the low denoise of 0.2. The image remains relatively unchanged.

![LATENT_UPSCALER_WF](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/6ae1779d-42ec-413e-8e44-1b9b8a1e2663)

## Training

### Upscaler v2.0

I decided to do some more research and change the network architecture alltogether. This one is just a bunch of `Conv2d` layers with an `Upsample` at the beginning, similar to before except I reduced the kernel size/padding and instead added more layers.

Trained for 1M iterations on DIV2K + Flickr2K. I changed to AdamW + L1 loss (from SGD and MSE loss) and added a `OneCycleLR` scheduler.

![loss](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/ca361dfd-7148-4b1b-bbf2-59151f8992cc)

### Upscaler v1.0 

This version was still relatively undertrained. Mostly a proof-of-concept.

Trained for 1M iterations on DIV2K + Flickr2K.

<details>
  <summary>Loss graphs for v1.0 models</summary>

  (Left is training loss, right is validation loss.)

  ![loss](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/edbc30b4-56b4-4b74-8c0b-3ab35916e963)

<details>
