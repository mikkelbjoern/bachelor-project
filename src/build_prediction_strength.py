from cProfile import label
from src.utils import (
    get_resnet_mixup_model,
    get_model_dir,
    ham10000_metadata,
    scientific_notation,
    short_to_full_name_dict,
    full_name_to_short_dict,
    HAM10000_DATA_FOLDER,
)
from src.config import resnet_mixup_id, only_lesion_id
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from fastai.interpret import ClassificationInterpretation


def build_prediction_strength():
    print("Building prediction strength...")

    # Read the validation results
    predictions = pd.read_csv(f"{get_model_dir(resnet_mixup_id)}/predictions.csv")

    # Merge with the metadata
    predictions = predictions.merge(ham10000_metadata, on="image_id", how="left")

    predictions["real_class"] = predictions.dx.map(lambda x: short_to_full_name_dict[x])

    predictions["correct"] = predictions.classification == predictions.real_class

    # Print the precision
    print(np.mean(predictions["correct"]))

    # Make a confusion matrix
    confusion_matrix = (
        predictions.groupby(["real_class", "classification"]).size().unstack()
    )

    # Normalize over the x axis
    confusion_matrix_normalized = confusion_matrix.div(
        confusion_matrix.sum(axis=1), axis=0
    )

    confusion_matrix_normalized.columns = confusion_matrix_normalized.columns.map(
        lambda x: full_name_to_short_dict[x]
    )

    confusion_matrix_normalized.index = confusion_matrix_normalized.index.map(
        lambda x: full_name_to_short_dict[x]
    )

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    p = sns.heatmap(
        confusion_matrix_normalized, annot=True, cmap="Blues", annot_kws={"size": 10}
    )
    p.set_xlabel("Predicted label")
    p.set_ylabel("True label")
    plt.savefig("confusion_matrix_seaborn.png")

    TINY_SIZE = (4, 4)
    only_melanoma = predictions[
        predictions["real_class"] == short_to_full_name_dict["mel"]
    ]
    mel_confusion_matrix = only_melanoma.groupby(["ruler", "correct"]).size().unstack()
    print(type(mel_confusion_matrix))
    # Heatmap of correct melanoma predictions sepearted on ruler vs. non-ruler
    plt.figure(figsize=TINY_SIZE)
    p = sns.heatmap(
        mel_confusion_matrix, annot=True, cmap="Blues", annot_kws={"size": 10}
    )
    p.set_ylabel("Has ruler")
    p.set_xlabel("Correctly classified")
    plt.savefig("mel_confusion_matrix_seaborn.png")

    # Make a chi-square test on the mel_confusion_matrix
    chi2, prop, dof, expected = stats.chi2_contingency(mel_confusion_matrix)
    print(prop)
    with open("p_mel.txt", "w") as f:
        f.write(f"{round(float(prop), 3)}")


    # Do a second plot where there is normalized over the x axis
    plt.figure(figsize=TINY_SIZE)
    p = sns.heatmap(
        mel_confusion_matrix.div(mel_confusion_matrix.sum(axis=1), axis=0),
        annot=True,
        cmap="Blues",
        annot_kws={"size": 10},
    )
    p.set_ylabel("Has ruler")
    p.set_xlabel("Correctly classified")
    plt.savefig("mel_confusion_matrix_seaborn_normalized.png")



