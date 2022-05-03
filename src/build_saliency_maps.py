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
        axs1.imshow(saliency_mean[0], cmap=plt.cm.hot)
        axs1.axis("off")
        axs1.set_title("Saliency Map")
        # axs1.title("Saliency map")
        axs2.imshow(color_image)
        axs2.imshow(saliency_mean[0], cmap=plt.cm.hot, alpha=0.6)
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

        


