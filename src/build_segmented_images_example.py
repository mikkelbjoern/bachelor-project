from src.utils import get_image_path, ham10000_metadata
import matplotlib.pyplot as plt
from PIL import Image

def build_segmented_images_examlpe():
    """
    Print a row of 5 sampled images examples and below them
    the segmented versions of those images (only_lesions)
    """
    print("Building segmented images example...")

    N = 5
    samlpe = ham10000_metadata.sample(N, random_state=42)
    # 2 rows, N columns
    plt.subplot(2, N, 1)

    # Set figsize
    plt.figure(figsize=(N * 2, 3))

    index = 0
    for _, row in samlpe.iterrows():
        image_path = get_image_path(row.image_id)
        only_lesion_path = get_image_path(row.image_id, 'only_lesions')
        image = Image.open(image_path)
        only_lesion = Image.open(only_lesion_path)
        plt.subplot(2, N, index + 1)
        plt.imshow(image)
        # Remove axes
        plt.axis("off")
        plt.subplot(2, N, index + 6)
        plt.imshow(only_lesion)
        plt.axis("off")
        index += 1

    plt.savefig('segmented_images_example.png', bbox_inches='tight')



