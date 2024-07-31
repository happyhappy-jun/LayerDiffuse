from typing import List, Optional, Union

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import \
    StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import \
    StableDiffusionXLImg2ImgPipeline
from diffusers.utils.torch_utils import randn_tensor

from LayerDiffuse.pipelines.sampler import KModel, sample_dpmpp_2m


class KDiffusionStableDiffusionXLPipeline(StableDiffusionXLImg2ImgPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_model = KModel(unet=kwargs['unet'])

    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        device = self.text_encoder.device
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2]

        pooled_prompt_embeds = None
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_input_ids = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # Only last pooler_output is needed
            pooled_prompt_embeds = prompt_embeds.pooler_output

            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        return prompt_embeds, pooled_prompt_embeds

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent: torch.FloatTensor = None,
            strength: float = 1.0,
            num_inference_steps: int = 25,
            guidance_scale: float = 5.0,
            batch_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        device = self.unet.device

        # Sigmas

        sigmas = self.k_model.get_sigmas_karras(int(num_inference_steps / strength))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)

        # Initial latents

        _, C, H, W = initial_latent.shape
        noise = randn_tensor((batch_size, C, H, W), generator=generator, device=device, dtype=self.unet.dtype)
        latents = initial_latent.to(noise) + noise * sigmas[0].to(noise)

        # Shape

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        add_time_ids = list((height, width) + (0, 0) + (height, width))
        add_time_ids = torch.tensor([add_time_ids], dtype=self.unet.dtype)
        add_neg_time_ids = add_time_ids.clone()

        # Batch

        latents = latents.to(device)
        add_time_ids = add_time_ids.repeat(batch_size, 1).to(device)
        add_neg_time_ids = add_neg_time_ids.repeat(batch_size, 1).to(device)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1).to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1).to(device)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(batch_size, 1).to(device)

        # Feeds

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}, ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                added_cond_kwargs={"text_embeds": negative_pooled_prompt_embeds, "time_ids": add_neg_time_ids},
            )
        )

        # Sample

        results = sample_dpmpp_2m(self.k_model, latents, sigmas, extra_args=sampler_kwargs, disable=False)

        return StableDiffusionXLPipelineOutput(images=results)
