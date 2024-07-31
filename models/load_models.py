import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file
from lib_layerdiffuse.vae import TransparentVAEDecoder, TransparentVAEEncoder


class ModelLoader:
    def __init__(self, sdxl_name):
        self.sdxl_name = sdxl_name
        self.tokenizer = None
        self.tokenizer_2 = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.vae = None
        self.unet = None

    def load_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.sdxl_name, subfolder="tokenizer")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(self.sdxl_name, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(self.sdxl_name, subfolder="text_encoder",
                                                          torch_dtype=torch.float16, variant="fp16")
        self.text_encoder_2 = CLIPTextModel.from_pretrained(self.sdxl_name, subfolder="text_encoder_2",
                                                            torch_dtype=torch.float16, variant="fp16")
        self.vae = AutoencoderKL.from_pretrained(self.sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16,
                                                 variant="fp16")
        self.unet = UNet2DConditionModel.from_pretrained(self.sdxl_name, subfolder="unet", torch_dtype=torch.float16,
                                                         variant="fp16")

    def apply_modifications(self, paths):
        sd_offset = load_file(paths['attn'])
        sd_origin = self.unet.state_dict()
        sd_merged = {k: sd_origin[k] + sd_offset[k] if k in sd_offset else sd_origin[k] for k in sd_origin.keys()}
        self.unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged

        transparent_encoder = TransparentVAEEncoder(paths['encoder'])
        transparent_decoder = TransparentVAEDecoder(paths['decoder'])
        return transparent_encoder, transparent_decoder
