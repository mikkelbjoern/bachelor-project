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

def build_near_neigh():
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

        with torch.no_grad():
            extraction_model.eval()
            return extraction_model(image_tensor)



    # ruler_sample = df_standard[df_standard.ruler == 1].sample(400)
    # no_ruler_sample = df_standard[df_standard.ruler == 0].sample(200)

    # Fasten pandas randomness with a seed
    np.random.seed(42)
    sample = df_standard.sample(101) # Set to 1500 for almost full test set
    ruler_sample = sample[sample.ruler == 1]
    no_ruler_sample = sample[sample.ruler == 0]

    features = {
        image_id: np.array(extract_features(image_id)[0]) for image_id in sample.image_id
    }
    has_ruler = {image_id: 1 for image_id in ruler_sample.image_id}
    has_ruler = {**has_ruler, **{image_id: 0 for image_id in no_ruler_sample.image_id}}

    # Get a random image from each of the two samples
    ruler_examples = ruler_sample.sample(10, random_state=1337).image_id

    home_path = os.path.expanduser("~")

    for im_index, image_id in enumerate(ruler_examples):
        image_vector = features[image_id]
        # Find the 5 nearest neighbors
        distances = {
            image_id: np.linalg.norm(image_vector - features[image_id])
            for image_id in features.keys()
        }
        nearest_neighbors = sorted(distances, key=distances.get)[1:6]
        # sorted(
        #     features.items(),
        #     key=lambda x: np.linalg.norm(x[1] - image_vector),
        #     reverse=True,
        # )[:5]
        # Plot the six images next to each other
        plt.figure(figsize=(14, 6), dpi=100)
        plt.subplot(im_index+1, 6, 1)
        plt.imshow(
            Image.open(f"{home_path}/kaggle-data/HAM10000/HAM10000_images/{image_id}.jpg")
        )
        plt.axis("off")
        plt.title(f"Query: {image_id}")
        for i, neighbor_id in enumerate(nearest_neighbors):
            plt.subplot(im_index+1, 6, i + 2)
            plt.imshow(
                Image.open(
                    f"{home_path}/kaggle-data/HAM10000/HAM10000_images/{neighbor_id}.jpg"
                )
            )
            # Remove the axis labels
            plt.axis("off")
            plt.title(
                f"{neighbor_id}\n{'Has ruler' if has_ruler[neighbor_id] else 'No ruler'}"
            )
    plt.savefig(f"examples.png")
