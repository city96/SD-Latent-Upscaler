# SD-Latent-Upscaler
Upscaling stable diffusion latents using a small neural network.

Very similar to my [latent interposer](https://github.com/city96/SD-Latent-Interposer/tree/main), this small model can be used to upscale latents in a way that doesn't ruin the image. I mostly explain some of the issues with upscaling latents in [this issue](https://github.com/city96/SD-Advanced-Noise/issues/1#issuecomment-1678193121). Think of this as an ESRGAN for latents, except severely undertrained.

**Currently, SDXL has some minimal hue shift issues.** Because of course it does.

## Installation

### ComfyUI

To install it, simply clone this repo to your custom_nodes folder using the following command: git clone https://github.com/city96/SD-Latent-Interposer custom_nodes/SD-Latent-Interposer.

Alternatively, you can download the [comfy_latent_upscaler.py](https://github.com/city96/SD-Latent-Upscaler/blob/main/comfy_latent_upscaler.py) file to your ComfyUI/custom_nodes folder as well. You may need to install hfhub using the command pip install huggingface-hub inside your venv.

If you need the model weights for something else, they are [hosted on HF](https://huggingface.co/city96/SD-Latent-Upscaler/tree/main) under the same Apache2 license as the rest of the repo.

### Auto1111

Currently not supported but it should be possible to use it at the hires-fix part.

### Usage/principle

Usage is fairly simple. You use it anywhere where you would upscale a latent. If you need a higher scale factor (e.g. x4), simply chain two of the upscalers.

![LATENT_UPSCALER_ANI](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/dc187631-fd94-445e-9f20-a5741091bb0e)

![LATENT_UPSCALER_COLOR](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/ec6997ce-664b-4956-a947-503b8b591f73)

## Training

### Interposer v1.0 

This current version is still relatively undertrained, as with the interposer. Mostly a proof-of-concept but it seems good enough as a base.

Trained for 1M iterations on DIV2K + Flickr2K.

(Left is training loss, right is validation loss.)
![loss](https://github.com/city96/SD-Latent-Upscaler/assets/125218114/edbc30b4-56b4-4b74-8c0b-3ab35916e963)
