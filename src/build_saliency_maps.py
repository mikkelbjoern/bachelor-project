from src.utils import (
    ham10000_metadata,
    DATA_FOLDER,
    get_resnet_mixup_model,
    HAM10000_DATA_FOLDER,
    short_to_full_name_dict,
)
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter, maximum_filter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_saliency_maps():
    print("Building saliency maps...")
    # Read the validation file names
    image_ids = []
    with open(f"{DATA_FOLDER}/valid_file_names.csv", "r") as f:
        for line in f:
            file_name = line.strip().split("/")[-1]
            image_id = file_name.split(".")[0]
            image_ids.append(image_id)

    # Find 3 pictures from the validation set with rulers
    with_rulers = ham10000_metadata[ham10000_metadata["ruler"] == 1.0]

    # Go through the validation set until we find 3 pictures with rulers
    print_image_ids = []
    check_index = 0
    while len(print_image_ids) < 8:
        if check_index >= len(with_rulers):
            print("Could not find 8 pictures with rulers")
            break
        if with_rulers.iloc[check_index]["image_id"] in image_ids:
            print_image_ids.append(with_rulers.iloc[check_index]["image_id"])
        check_index += 1

    learn = get_resnet_mixup_model()
    learn.no_logging()
    learn.model.requires_grad_()
    image_paths = [
        HAM10000_DATA_FOLDER + "/HAM10000_images/" + iid + ".jpg"
        for iid in print_image_ids
    ]

    predictions = [learn.predict(img_path, with_input=True) for img_path in image_paths]
    
    latex_appendix = ""
    for i, pred in enumerate(predictions):
        img, classification, class_index, pred_tensor = pred
        # Turn img to tensor
        img = img.float()
        img = img.to(DEVICE)
        img = img.unsqueeze(0)
        img.requires_grad_()

        scores = learn.model(img)
        score = scores[0][class_index]
        score.backward()
        saliency_mean = torch.mean(img.grad.data.abs(), dim=1)
        # Plot the saliency map
        color_image = Image.open(image_paths[i]).resize((450, 450))

        # Get the real class of the image
        real_class = ham10000_metadata[
            ham10000_metadata["image_id"] == print_image_ids[i]
        ]["dx"].values[0]
        real_class = short_to_full_name_dict[real_class]

        fig, (axs0, axs1, axs2) = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
        fig.suptitle(
            f"{real_class} lesion classified as {classification}"
            if classification != real_class
            else f"Correctly classified {real_class}",
            fontsize=16,
        )
        axs0.imshow(color_image)
        axs0.axis("off")
        axs0.set_title("Original Image")
        # axs0.title("original photo")

        if str(DEVICE).startswith("cuda"):
            saliency_mean = saliency_mean.cpu().numpy()
        
        axs1.imshow(saliency_mean[0], cmap=plt.cm.hot)
        axs1.axis("off")
        axs1.set_title("Saliency Map")
        # Multiple the saliency map by the original by element-wise multiplication
        # Change saliency map to numpy array
        saliency_mean = saliency_mean[0]

        # Apply a uniform filter to the saliency map
        footprint = np.ones((12, 12)) * 0.4
        footprint[1:-1, 1:-1] += 0.3
        footprint[3:-3, 3:-3] += 0.2
        footprint[5:-5, 5:-5] = 1
        saliency_mean = maximum_filter(saliency_mean, footprint=footprint)
        saliency_mean = maximum_filter(saliency_mean, size=3)
        saliency_mean = uniform_filter(saliency_mean, size=12)

        # Stretch the saliency map values to the 0-1 range 
        saliency_mean = (saliency_mean - np.min(saliency_mean)) / (
            np.max(saliency_mean) - np.min(saliency_mean)
        )


        heat_focused_image = color_image.copy()
        # Turn image into numpy array
        heat_focused_image = np.array(heat_focused_image)
        # Multiply the saliency map by the original by element-wise multiplication
        # on each of the 3 channels
        heat_focused_image[:, :, 0] = heat_focused_image[:, :, 0] * saliency_mean
        heat_focused_image[:, :, 1] = heat_focused_image[:, :, 1] * saliency_mean
        heat_focused_image[:, :, 2] = heat_focused_image[:, :, 2] * saliency_mean
        # Turn image back into PIL image
        heat_focused_image = Image.fromarray(heat_focused_image)


        axs2.imshow(heat_focused_image, cmap=plt.cm.hot)
        axs2.axis("off")
        axs2.set_title("Saliency Map Overlay")

        plt.savefig(f"overview_map_{i}.png")
        plt.clf()

        # Write a figure to the appendix
        latex_safe_id = print_image_ids[i].replace("_", "\\_")
        latex_appendix += f"""\\begin{{figure}}[h]
        \includegraphics[
            width=\\textwidth,
            height=\\textheight,
            keepaspectratio=true,
            angle=0,
            clip=false
        ]{{{f"build/saliency_maps/overview_map_{i}.png"}}}
        \caption{{Saliency map of {latex_safe_id}.}}
        \end{{figure}}
        
        """
    with open("saliency_maps_appendix.tex", "w") as f:
        f.write(latex_appendix)

        


