from torch import nn
import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from src.utils import (
    get_resnet_mixup_model,
    get_model_dir,
    ham10000_metadata,
    full_name_to_short_dict,
)
import src.config as config
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bening_or_malignant_dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_near_neigh():
    print("Building near neigh plot...")
    model_dir = get_model_dir(config.resnet_mixup_id)

    df_standard = pd.read_csv(f"{model_dir}/predictions.csv").merge(
        ham10000_metadata, on="image_id"
    )
    df_standard["classification_short"] = df_standard.classification.map(
        lambda x: full_name_to_short_dict[x]
    )

    learn = get_resnet_mixup_model()
    model = learn.model
    # list(model.modules())[0] has length 2, I think the second part is
    # the layer that is adapting to the specific classes.
    # We are interested in the first part - the convolutional layers.

    # Make a model that only extracts the features
    extraction_model = nn.Sequential(
        # *(list(model.children())[:-1])
        *(list(model.children())[:-1])
    )

    def extract_features(image_id):
        home_path = os.path.expanduser("~")
        image = Image.open(
            f"{home_path}/kaggle-data/HAM10000/HAM10000_images/{image_id}.jpg"
        )
        image = image.convert("RGB")
        image = image.resize((450, 450))
        image_tensor = transforms.PILToTensor()(image)
        image_tensor = image_tensor.float()
        image_tensor = image_tensor.unsqueeze(0)
        if str(DEVICE).startswith("cuda"):
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            extraction_model.eval()
            res = extraction_model(image_tensor)
            if str(DEVICE).startswith("cuda"):
                res = res.cpu()
            return res

    # ruler_sample = df_standard[df_standard.ruler == 1].sample(400)
    # no_ruler_sample = df_standard[df_standard.ruler == 0].sample(200)

    # Fasten pandas randomness with a seed
    np.random.seed(42)
    sample = df_standard.sample(1500)  # Set to 1500 for almost full test set
    ruler_sample = sample[sample.ruler == 1]
    no_ruler_sample = sample[sample.ruler == 0]

    features = {
        image_id: np.array(extract_features(image_id)[0])
        for image_id in sample.image_id
    }
    has_ruler = {image_id: 1 for image_id in ruler_sample.image_id}
    has_ruler = {**has_ruler, **{image_id: 0 for image_id in no_ruler_sample.image_id}}

    # Get a random image from each of the two samples
    ruler_examples = ruler_sample.sample(4, random_state=1337).image_id

    home_path = os.path.expanduser("~")

    NNs = 5
    fig, ax = plt.subplots(
        len(ruler_examples), NNs + 1, figsize=(NNs*2+0.5, len(ruler_examples) * 1.5 + 1)
    )

    for im_index, image_id in enumerate(ruler_examples):
        query_ax = ax[im_index, 0]
        image_vector = features[image_id]
        # Find the 5 nearest neighbors
        distances = {
            image_id: np.linalg.norm(image_vector - features[image_id])
            for image_id in features.keys()
        }
        nearest_neighbors = sorted(distances, key=distances.get)[1:NNs + 1]
        # sorted(
        #     features.items(),
        #     key=lambda x: np.linalg.norm(x[1] - image_vector),
        #     reverse=True,
        # )[:5]
        # Plot the six images next to each other
        query_ax.imshow(
            Image.open(
                f"{home_path}/kaggle-data/HAM10000/HAM10000_images/{image_id}.jpg"
            )
        )
        query_ax.axis("off")
        query_ax.set_title(f"Query: {image_id}")
        for i, neighbor_id in enumerate(nearest_neighbors):
            local_ax = ax[im_index, i+1]
            local_ax.imshow(
                Image.open(
                    f"{home_path}/kaggle-data/HAM10000/HAM10000_images/{neighbor_id}.jpg"
                )
            )
            # Remove the axis labels
            local_ax.axis("off")
            local_ax.set_title(
                f"{neighbor_id}\n{'Has ruler' if has_ruler[neighbor_id] else 'No ruler'}"
            )

    fig.savefig("examples.png")
    
