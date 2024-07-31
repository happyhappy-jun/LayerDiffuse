import os
import numpy as np
from PIL import Image
from LayerDiffuse.models.download import download_models
from LayerDiffuse.models.load_models import ModelLoader
from LayerDiffuse.pipelines.pipeline import KDiffusionStableDiffusionXLPipeline
from LayerDiffuse.utils.image_utils import get_binary_mask, save_image
from LayerDiffuse.utils.memory_management import load_models_to_gpu, unload_all_models

os.environ['HF_HOME'] = 'D:/hf_home'

SDXL_NAME = 'SG161222/RealVisXL_V4.0'
DEFAULT_NEGATIVE = 'face asymmetry, eyes asymmetry, deformed eyes, open mouth'


class ImageGenerator:
    def __init__(self, sdxl_name=SDXL_NAME, default_negative=DEFAULT_NEGATIVE):
        self.sdxl_name = sdxl_name
        self.default_negative = default_negative
        self.model_loader = None
        self.pipeline = None
        self.transparent_encoder = None
        self.transparent_decoder = None

        # Initialize models and pipeline
        self._initialize_models()

    def _initialize_models(self):
        # Download models
        paths = download_models()

        # Load models
        self.model_loader = ModelLoader(self.sdxl_name)
        self.model_loader.load_models()
        self.transparent_encoder, self.transparent_decoder = self.model_loader.apply_modifications(paths)

        # Set up pipeline
        self.pipeline = KDiffusionStableDiffusionXLPipeline(
            vae=self.model_loader.vae,
            text_encoder=self.model_loader.text_encoder,
            tokenizer=self.model_loader.tokenizer,
            text_encoder_2=self.model_loader.text_encoder_2,
            tokenizer_2=self.model_loader.tokenizer_2,
            unet=self.model_loader.unet,
            scheduler=None
        )

        # Load models to GPU
        load_models_to_gpu([
            self.model_loader.vae,
            self.model_loader.text_encoder,
            self.model_loader.text_encoder_2,
            self.transparent_encoder,
            self.transparent_decoder,
            self.model_loader.unet
        ])

    def generate_images(self, image_path, prompt, guidance_scale=7.0, strength=0.7, num_inference_steps=25):
        # Load and prepare initial latent
        initial_image = [np.array(Image.open(image_path))]
        initial_latent = self.transparent_encoder(self.model_loader.vae,
                                                  initial_image) * self.model_loader.vae.config.scaling_factor
        initial_latent = initial_latent.to(dtype=self.model_loader.unet.dtype, device=self.model_loader.unet.device)

        # Encode prompts
        positive_cond, positive_pooler = self.pipeline.encode_cropped_prompt_77tokens(prompt)
        negative_cond, negative_pooler = self.pipeline.encode_cropped_prompt_77tokens(self.default_negative)

        # Generate images
        latents = self.pipeline(
            initial_latent=initial_latent,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            batch_size=1,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler
        )

        # Prepare latent for decoding
        image_latent = latents.images
        image_latent = image_latent.to(dtype=self.model_loader.vae.dtype,
                                       device=self.model_loader.vae.device) / self.model_loader.vae.config.scaling_factor
        final_results, visualizations = self.transparent_decoder(self.model_loader.vae, image_latent)

        return final_results, visualizations

    def save_images(self, final_results, visualizations, output_dir='./imgs/outputs'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save results
        for i, image in enumerate(final_results):
            output_image_path = os.path.join(output_dir, f'i2i_{i}_transparent.png')
            output_mask_path = os.path.join(output_dir, f'i2i_{i}_transparent_mask.png')
            Image.fromarray(image).save(output_image_path, format='PNG')
            binary_mask = get_binary_mask(image)
            save_image(binary_mask, output_mask_path)

        for i, image in enumerate(visualizations):
            output_vis_path = os.path.join(output_dir, f'i2i_{i}_visualization.png')
            Image.fromarray(image).save(output_vis_path, format='PNG')

    def __del__(self):
        # Unload all models when the instance is deleted
        unload_all_models()
