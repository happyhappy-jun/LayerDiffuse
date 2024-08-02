from PIL import Image, ImageOps

from LayerDiffuse.pipelines.transparent_image_generator import TransparentImageGenerator


def pad_to_square(image, target_size=None):
    # Get the dimensions of the image
    width, height = image.size
    if target_size is not None:
        max_dim = target_size
    else:
        if width == height:
            return image
        max_dim = max(width, height)

    pad_width = (max_dim - width) // 2
    pad_height = (max_dim - height) // 2

    # Add padding to make the image square
    padded_image = ImageOps.expand(image,
                                   (pad_width, pad_height, max_dim - width - pad_width, max_dim - height - pad_height),
                                   fill=(0, 0, 0, 0))

    return padded_image


def main():
    # Create an instance of ImageGenerator
    image_generator = TransparentImageGenerator()

    # Generate images from an input image and prompt
    image = Image.open("/home/junyoon/sam-aug/LayerDiffuse/imgs/inputs/ILSVRC2012_test_00000003.jpg").convert("RGBA")
    alpha_channel = Image.open("/home/junyoon/sam-aug/data/cascade_psp/DUTS-TE/ILSVRC2012_test_00000003.png").convert("L")
    image.putalpha(alpha_channel)
    image = pad_to_square(image, target_size=1024)
    final_results, visualizations = image_generator.generate_images(
        image=image,
        prompt='black dog, high quality'
    )

    # Save the generated images and masks
    image_generator.save_images(final_results, visualizations)


if __name__ == "__main__":
    main()
