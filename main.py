from LayerDiffuse.pipelines.transparent_image_generator import TransparentImageGenerator


def main():
    # Create an instance of ImageGenerator
    image_generator = TransparentImageGenerator()

    # Generate images from an input image and prompt
    final_results, visualizations = image_generator.generate_images(
        image_path='./imgs/inputs/causal_cut.png',
        prompt='a handsome man with curly hair, high quality'
    )

    # Save the generated images and masks
    image_generator.save_images(final_results, visualizations)


if __name__ == "__main__":
    main()
