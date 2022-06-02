from src.utils import get_image_path, ham10000_metadata
import pandas as pd
import matplotlib.pyplot as plt

def build_example_images():
    """
    Makes a row with lesion images with ther corresponding labels.
    """
    print("Building example images...")

    example_images = []
    # Extract an image from each class
    for class_name in ham10000_metadata.dx.unique():
        # Find an image with the corresponding class
        image_path = get_image_path(ham10000_metadata[ham10000_metadata.dx == class_name].image_id.values[0])
        example_images.append((image_path, class_name))

    # Add an extra mel and nv image
    for class_name in ["mel", "nv"]:
        image_path = get_image_path(ham10000_metadata[ham10000_metadata.dx == class_name].image_id.values[1])
        example_images.append((image_path, class_name))

    # Make a plot with the images (3 rows)
    plt.figure(figsize=(8, 6))
    for i, (image_path, class_name) in enumerate(example_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(plt.imread(image_path))
        # Print the class name bold if the image is malignant (mel, akiec, bcc)
        if class_name in ["mel", "akiec", "bcc"]:
            plt.title(f"{class_name}", fontweight="bold")
        else:
            plt.title(f"{class_name}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("example_images.png")
