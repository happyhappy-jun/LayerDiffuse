import os


from LayerDiffuse.models.download import download_models
from LayerDiffuse.pipelines.pipeline import KDiffusionStableDiffusionXLPipeline

os.environ['HF_HOME'] = 'D:/hf_home'

import numpy as np
import safetensors.torch as sf
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from LayerDiffuse.utils import memory_management
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder

# Load models

sdxl_name = 'SG161222/RealVisXL_V4.0'
tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

# This negative prompt is suggested by RealVisXL_V4 author
# See also https://huggingface.co/SG161222/RealVisXL_V4.0
# Note that in A111's normalization, a full "(full sentence)" is equal to "full sentence"
# so we can just remove SG161222's braces

default_negative = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Download Mode

model_path = download_models()

# Modify

sd_offset = sf.load_file(model_path["attn"])
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {}

for k in sd_origin.keys():
    if k in sd_offset:
        sd_merged[k] = sd_origin[k] + sd_offset[k]
    else:
        sd_merged[k] = sd_origin[k]

unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys, k

transparent_encoder = TransparentVAEEncoder(model_path["encoder"])
transparent_decoder = TransparentVAEDecoder(model_path["decoder"])

# Pipelines

pipeline = KDiffusionStableDiffusionXLPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)


@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


with torch.inference_mode():
    guidance_scale = 7.0

    rng = torch.Generator(device=memory_management.gpu).manual_seed(12345)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(
        'glass bottle, high quality'
    )

    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(default_negative)

    memory_management.load_models_to_gpu([unet])
    initial_latent = torch.zeros(size=(1, 4, 144, 112), dtype=unet.dtype, device=unet.device)
    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=25,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=guidance_scale,
    ).images

    memory_management.load_models_to_gpu([vae, transparent_decoder, transparent_encoder])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    result_list, vis_list = transparent_decoder(vae, latents)

    for i, image in enumerate(result_list):
        Image.fromarray(image).save(f'./imgs/outputs/t2i_{i}_transparent.png', format='PNG')

    for i, image in enumerate(vis_list):
        Image.fromarray(image).save(f'./imgs/outputs/t2i_{i}_visualization.png', format='PNG')
