from cProfile import label
from src.utils import (
    get_resnet_mixup_model,
    get_model_dir,
    ham10000_metadata,
    scientific_notation,
    short_to_full_name_dict,
    full_name_to_short_dict,
    HAM10000_DATA_FOLDER,
    bening_or_malignant_dict,
)
from src.config import resnet_mixup_id, only_lesion_id
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    #### Comparison of performance on cases with and without a ruler ####
    plt.clf()

    ruler_confusion_matrix = predictions.groupby(["ruler", "correct"]).size().unstack()
    ruler_plot = sns.heatmap(ruler_confusion_matrix, annot=True, cmap="Blues", fmt="d")
    ruler_plot.set_ylabel("Has ruler")
    ruler_plot.set_xlabel("Correctly classified")
    plt.savefig("ruler_confusion_matrix_seaborn.png")

    # Normalize over the x axis
    plt.clf()
    ruler_confusion_matrix_normalized = ruler_confusion_matrix.div(
        ruler_confusion_matrix.sum(axis=1), axis=0
    )
    ruler_plot_normalized = sns.heatmap(
        ruler_confusion_matrix_normalized, annot=True, cmap="Blues"
    )
    ruler_plot_normalized.set_ylabel("Has ruler")
    ruler_plot_normalized.set_xlabel("Correctly classified")
    plt.savefig("ruler_confusion_matrix_seaborn_normalized.png")

    # Make a chi-square test on the ruler_confusion_matrix
    chi2, prop, dof, expected = stats.chi2_contingency(ruler_confusion_matrix)
    with open("p_ruler.txt", "w") as f:
        f.write(f"{round(float(prop), 3)}")

    # Do seperate plots and tests for the malignant and benign cases
    predictions["benign_or_malignant"] = predictions.dx.map(
        lambda x: bening_or_malignant_dict[x]
    )

    benign_predictions = predictions[predictions["benign_or_malignant"] == "benign"]
    malignant_predictions = predictions[
        predictions["benign_or_malignant"] == "malignant"
    ]

    benign_confusion_matrix = (
        benign_predictions.groupby(["ruler", "correct"]).size().unstack()
    )
    malignant_confusion_matrix = (
        malignant_predictions.groupby(["ruler", "correct"]).size().unstack()
    )

    benign_confusion_matrix_normalized = benign_confusion_matrix.div(
        benign_confusion_matrix.sum(axis=1), axis=0
    )
    malignant_confusion_matrix_normalized = malignant_confusion_matrix.div(
        malignant_confusion_matrix.sum(axis=1), axis=0
    )

    plt.clf()
    benign_plot = sns.heatmap(
        benign_confusion_matrix, annot=True, cmap="Blues", fmt="d"
    )
    benign_plot.set_ylabel("Has ruler")
    benign_plot.set_xlabel("Correctly classified")
    plt.savefig("benign_confusion_matrix_seaborn.png")

    plt.clf()
    malignant_plot = sns.heatmap(
        malignant_confusion_matrix, annot=True, cmap="Blues", fmt="d"
    )
    malignant_plot.set_ylabel("Has ruler")
    malignant_plot.set_xlabel("Correctly classified")
    plt.savefig("malignant_confusion_matrix_seaborn.png")

    plt.clf()
    benign_plot_normalized = sns.heatmap(
        benign_confusion_matrix_normalized, annot=True, cmap="Blues"
    )
    benign_plot_normalized.set_ylabel("Has ruler")
    benign_plot_normalized.set_xlabel("Correctly classified")
    plt.savefig("benign_confusion_matrix_seaborn_normalized.png")

    plt.clf()
    malignant_plot_normalized = sns.heatmap(
        malignant_confusion_matrix_normalized, annot=True, cmap="Blues"
    )
    malignant_plot_normalized.set_ylabel("Has ruler")
    malignant_plot_normalized.set_xlabel("Correctly classified")
    plt.savefig("malignant_confusion_matrix_seaborn_normalized.png")

    # Make a chi-square test on the malignant_confusion_matrix
    chi2, prop, dof, expected = stats.chi2_contingency(malignant_confusion_matrix)
    with open("p_malignant.txt", "w") as f:
        f.write(f"{round(float(prop), 3)}")

    # Make a chi-square test on the benign_confusion_matrix
    chi2, prop, dof, expected = stats.chi2_contingency(benign_confusion_matrix)
    with open("p_benign.txt", "w") as f:
        f.write(f"{round(float(prop), 5)}")

    ### Making plots of where the wrong predictions go ###
    # # Show what classes the false classified beningn lesions are from
    # false_benign_predictions = benign_predictions[
    #     (benign_predictions["correct"] == False) & (benign_predictions["ruler"] == True)
    # ]

    # classi_count_false = false_benign_predictions.groupby(["classification"]).size()
    # print(classi_count_false)

    # # Plot the count
    # plt.clf()
    # plt.figure(figsize=(8, 5))
    # count_plot = plt.bar(
    #     [full_name_to_short_dict[x] for x in classi_count_false.index],
    #     classi_count_false.values,
    #     # Color blue for benign and red for malignant
    #     color=[
    #         "blue" if bening_or_malignant_dict[full_name_to_short_dict[x]] == "benign" else "red"
    #         for x in classi_count_false.index
    #     ],
    # )
    # # Show benign in legend
    # benign_blue_patch = mpatches.Patch(color="blue", label="Benign lesion class")
    # # Show malignant in legend
    # malignant_red_patch = mpatches.Patch(color="red", label="Malignant lesion class")

    # plt.legend(handles=[benign_blue_patch, malignant_red_patch])
    # plt.savefig("false_classification_count.png")


    # # Make the same plot with falsely classified beningn lesions but also with data
    # # without rulers
    # false_benign_predictions_all = benign_predictions[
    #     benign_predictions["correct"] == False
    # ]

    # classi_count_false_all = false_benign_predictions_all.groupby(
    #     ["classification"]
    # ).size()
    # print(classi_count_false_all)

    # plt.clf()
    # plt.figure(figsize=(8, 5))
    # count_plot = plt.bar(
    #     [full_name_to_short_dict[x] for x in classi_count_false_all.index],
    #     classi_count_false_all.values,
    #     # Color blue for benign and red for malignant
    #     color=[
    #         "blue" if bening_or_malignant_dict[full_name_to_short_dict[x]] == "benign" else "red"
    #         for x in classi_count_false_all.index
    #     ],
    # )
    # # Show benign in legend
    # benign_blue_patch = mpatches.Patch(color="blue", label="Benign lesion class")
    # # Show malignant in legend
    # malignant_red_patch = mpatches.Patch(color="red", label="Malignant lesion class")

    # plt.legend(handles=[benign_blue_patch, malignant_red_patch])
    # plt.savefig("false_classification_count_all.png")

