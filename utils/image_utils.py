import numpy as np
import torch
from PIL import Image


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


def get_alpha_channel(image) -> np.ndarray:
    # Ensure image is in RGBA format
    image = Image.fromarray(image).convert("RGBA")

    # Extract the alpha channel
    alpha_channel = image.split()[-1]

    # Convert the alpha channel to a numpy array
    alpha_channel = np.array(alpha_channel)

    return alpha_channel


def get_binary_mask(image) -> np.ndarray:
    alpha_channel = get_alpha_channel(image)
    # Create a binary mask where non-transparent pixels are 255 (white) and transparent pixels are 0 (black)
    binary_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)
    return binary_mask


def save_image(image_array: np.ndarray, output_path) -> None:
    # Convert numpy array to PIL Image and save it
    image = Image.fromarray(image_array)
    image.save(output_path)
