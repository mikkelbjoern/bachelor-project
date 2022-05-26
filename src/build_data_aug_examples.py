from re import I

from torch import seed
from src.utils import HAM10000_DATA_FOLDER, ham10000_metadata
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def build_data_aug_examples():
    """
    Find a few images and show different kinds of data augmentation on them.
     * MixUp where two images are taken a mean of
     * Random crop where between 75% and 100% of the image is cropped out
     * Dihedral flip where the image is rotated by 90, 180, 270 degrees and possibly flipped
    """
    print("Building data augmentation examples...")

    # Choose some 2 images to do examples on
    sample = ham10000_metadata.sample(n=2, random_state=42)
    image_A_id = sample.iloc[0]["image_id"]
    image_B_id = sample.iloc[1]["image_id"]

    image_A_unscaled = Image.open(
        f"{HAM10000_DATA_FOLDER}/HAM10000_images/{image_A_id}.jpg"
    )
    image_B_unscaled = Image.open(
        f"{HAM10000_DATA_FOLDER}/HAM10000_images/{image_B_id}.jpg"
    )

    # Save the images to disk
    image_A_unscaled.save(f"image_A_unscaled.jpg")
    image_B_unscaled.save(f"image_B_unscaled.jpg")

    # Scale the images to be be 450x450
    image_A = image_A_unscaled.resize((450, 450))
    image_B = image_B_unscaled.resize((450, 450))

    # Save the images to disk
    image_A.save(f"image_A.jpg")
    image_B.save(f"image_B.jpg")

    ####### MixUp #######
    print(" > Doing MixUp...")
    # Use blend to mix the two images
    image_A_mixup = Image.blend(image_A, image_B, 0.5)
    image_A_mixup.save(f"image_A_mixup.jpg")

    ####### Random crop #######
    print(" > Doing Random crop...")
    # Crop the image
    # Choose a random crop size
    # Set np random seed
    np.random.seed(42)

    crop_size = (np.random.randint(75, 100) / 100)**0.5 * 450
    x, y = np.random.randint(0, 450 - crop_size, 2)
    image_A_crop = image_A.crop((x, y, x + crop_size, y + crop_size))
    image_A_crop.save(f"image_A_crop.jpg")

    image_B_crop = image_B.crop((x, y, x + crop_size, y + crop_size))
    image_B_crop.save(f"image_B_crop.jpg")

    ####### Dihedral flip #######
    print(" > Doing Dihedral flip...")
    # Flip the image
    image_A_flip = image_A.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotate it by 90, 180, 270 degrees
    for angle in [90, 180, 270]:
        image_A_rotate = image_A.rotate(angle)
        image_A_rotate.save(f"image_A_rotate_{angle}.jpg")
        image_A_flip_rotate = image_A_flip.rotate(angle)
        image_A_flip_rotate.save(f"image_A_flip_rotate_{angle}.jpg")

    # Make a subplot of all the dihedral flips and rotations
    # rotations along the x axis and flips along the y axis
    # Set these settings
    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    fig.subplots_adjust(
        top=0.954,
        bottom=0.049,
        left=0.043,
        right=0.987,
        hspace=0.07,
        wspace=0.094,
    )

    for flipped in [0, 1]:
        for angle in [0, 90, 180, 270]:
            image = image_A.copy()
            if flipped == 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            image = image.rotate(angle)
            ax[flipped, angle // 90].imshow(image)
            if flipped == 0:
                ax[flipped, angle // 90].set_title(f"Original rotated {angle} degrees")
            else:
                ax[flipped, angle // 90].set_title(f"Flipped rotated {angle} degrees")
    plt.tight_layout()
    plt.savefig("dihedral_flip_examples.png", dpi=100)
