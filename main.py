import os

import numpy as np
from PIL import Image

from LayerDiffuse.utils.image_utils import get_binary_mask, save_image
from models.download import download_models
from models.load_models import ModelLoader
from pipelines.pipeline import KDiffusionStableDiffusionXLPipeline
from utils.memory_management import load_models_to_gpu, unload_all_models

os.environ['HF_HOME'] = 'D:/hf_home'

SDXL_NAME = 'SG161222/RealVisXL_V4.0'
DEFAULT_NEGATIVE = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'


def main():
    # Start recording memory snapshot history
    # Download models
    paths = download_models()

    # Load models
    model_loader = ModelLoader(SDXL_NAME)
    model_loader.load_models()
    transparent_encoder, transparent_decoder = model_loader.apply_modifications(paths)

    # Set up pipeline
    pipeline = KDiffusionStableDiffusionXLPipeline(
        vae=model_loader.vae,
        text_encoder=model_loader.text_encoder,
        tokenizer=model_loader.tokenizer,
        text_encoder_2=model_loader.text_encoder_2,
        tokenizer_2=model_loader.tokenizer_2,
        unet=model_loader.unet,
        scheduler=None
    )

    # Load and prepare initial latent
    initial_image = [np.array(Image.open('./imgs/inputs/causal_cut.png'))]
    initial_latent = transparent_encoder(model_loader.vae, initial_image) * model_loader.vae.config.scaling_factor
    initial_latent = initial_latent.to(dtype=model_loader.unet.dtype, device=model_loader.unet.device)

    # Load necessary models to GPU
    load_models_to_gpu([model_loader.text_encoder, model_loader.text_encoder_2])

    # Encode prompts
    positive_cond, positive_pooler = pipeline.encode_cropped_prompt_77tokens(
        'a handsome man with curly hair, high quality'
    )
    negative_cond, negative_pooler = pipeline.encode_cropped_prompt_77tokens(DEFAULT_NEGATIVE)

    # Unload text encoders and load VAE and Transparent Encoder/Decoder to GPU
    load_models_to_gpu([model_loader.vae, transparent_encoder, transparent_decoder])

    # Prepare initial latent in the right device
    initial_latent = initial_latent.to(dtype=model_loader.unet.dtype, device=model_loader.unet.device)

    # Unload VAE and load UNET to GPU
    load_models_to_gpu([model_loader.unet])

    # Generate images
    latents = pipeline(
        initial_latent=initial_latent,
        strength=0.7,
        num_inference_steps=25,
        guidance_scale=7.0,
        batch_size=1,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler
    )

    # Unload UNET and load VAE and Transparent Decoder to GPU for final decoding
    load_models_to_gpu([model_loader.vae, transparent_decoder])
    image_latent = latents.images
    image_latent = image_latent.to(dtype=model_loader.vae.dtype,
                                   device=model_loader.vae.device) / model_loader.vae.config.scaling_factor
    final_results, visualizations = transparent_decoder(model_loader.vae, image_latent)

    # Save results
    for i, image in enumerate(final_results):
        Image.fromarray(image).save(f'./imgs/outputs/i2i_{i}_transparent.png', format='PNG')
        binary_mask = get_binary_mask(image)
        save_image(binary_mask, f'./imgs/outputs/i2i_{i}_transparent_mask.png')

    for i, image in enumerate(visualizations):
        Image.fromarray(image).save(f'./imgs/outputs/i2i_{i}_visualization.png', format='PNG')

    unload_all_models()


if __name__ == "__main__":
    main()
