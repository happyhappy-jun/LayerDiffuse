
from LayerDiffuse.lib_layerdiffuse.utils import download_model


def download_models():
    models = {
        'attn': 'https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_attn.safetensors',
        'encoder': 'https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_encoder.safetensors',
        'decoder': 'https://huggingface.co/lllyasviel/LayerDiffuse_Diffusers/resolve/main/ld_diffusers_sdxl_vae_transparent_decoder.safetensors'
    }
    local_paths = {}
    for name, url in models.items():
        local_path = f'./models/{name}.safetensors'
        local_paths[name] = download_model(url, local_path)
    return local_paths