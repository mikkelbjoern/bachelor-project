import os
from math import floor, log10
import pandas as pd
import src.config as config

# Find the image folder one level up from this file
IMAGE_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../images"

DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../data"

MODEL_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/../models"
# Check that the model folder exists, and if not, create it
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Find the data folder in the ~/kaggle-data/HAM10000
_home = os.path.expanduser("~")
HAM10000_DATA_FOLDER = _home + "/kaggle-data/HAM10000"


def bmatrix(a, format=None):
    """Returns a LaTeX bmatrix
    Source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if format is None:
        format = lambda x: x
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    numbers = [[format(x) for x in l.split()] for l in lines]
    rv += ["  " + " & ".join(n) + r"\\" for n in numbers]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)


def scientific_notation(x, precision=2):
    """Returns a string representation of a number in scientific notation.
    Source: https://stackoverflow.com/questions/3410976/how-to-format-numbers-in-python-with-significant-figures-and-without-exponential
    Modified to return LaTeX string.
    """
    if x == 0:
        return "0"
    elif x < 0:
        return "-" + scientific_notation(-x)
    else:
        exponent = int(floor(log10(x)))
        mantissa = x / 10**exponent
        return f"{round(mantissa, precision)} \cdot 10^{{{exponent}}}"


# Load the data
confounders = pd.read_csv(DATA_FOLDER + "/confounder_labels.csv")
labels = pd.read_csv(HAM10000_DATA_FOLDER + "/HAM10000_metadata.csv")
# Renaming to match the two datasets
confounders = confounders.rename(columns={"image": "image_id"})

# Merge the two datasets
ham10000_metadata = pd.merge(confounders, labels, on="image_id")


def get_model_dir(model_id):
    """Returns the folder for a model."""
    # Go through the model dir and find a dir containing the model_id
    for dir_name in os.listdir(MODEL_FOLDER):
        if model_id in dir_name:
            return MODEL_FOLDER + "/" + dir_name

    # If we get here, we didn't find the model
    hpc_command = f"cd {MODEL_FOLDER} && dtuhpc download {model_id[:8]}"
    raise ValueError(
        f"Could not find model {model_id}, please try to run:\n{hpc_command}"
    )


def get_resnet_mixup_model():
    """Returns the RESNET18-mixup model."""
    from models.resnet_mixup.ham10000_resnet import learn as learn_resnet_mixup

    model_dir = get_model_dir(config.resnet_mixup_id)
    learn_resnet_mixup.load(model_dir + "/models/model_resnet18")
    return learn_resnet_mixup


def get_only_lesions_model():
    from models.only_lesions.only_lesions import learn as learn_only_lesions

    model_dir = get_model_dir(config.only_lesions_id)
    learn_only_lesions.load(model_dir + "/models/model_resnet18")
    return learn_only_lesions


short_to_full_name_dict = {
    "akiec": "Bowen's disease",  # very early form of skin cancer
    "bcc": "basal cell carcinoma",  # basal-cell cancer or white skin cancer
    "bkl": "benign keratosis-like lesions",  # non-cancerous skin tumour
    "df": "dermatofibroma",  # non-cancerous rounded bumps
    "mel": "melanoma",  # black skin cancer
    "nv": "melanocytic nevi",  # mole non-cancerous
    "vasc": "vascular lesions",  # skin condition
}

full_name_to_short_dict = {v: k for k, v in short_to_full_name_dict.items()}
