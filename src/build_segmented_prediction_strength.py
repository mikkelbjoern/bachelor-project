from matplotlib.ft2font import BOLD
from src.utils import (
    get_model_dir,
    ham10000_metadata,
    short_to_full_name_dict,
    full_name_to_short_dict,
    bening_or_malignant_dict,
    get_image_path,
    calculate_metrics,
)
from PIL import Image
from src.config import only_lesion_id, resnet_mixup_id
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as stats
from fastai.interpret import ClassificationInterpretation


def build_segmented_prediction_strength():
    print("Building segmented prediction strength...")

    # Read the validation results
    predictions = pd.read_csv(f"{get_model_dir(only_lesion_id)}/predictions.csv")
    only_lesion_predictions = pd.read_csv(
        f"{get_model_dir(only_lesion_id)}/only_lesions_predictions.csv"
    )

    # Merge with the metadata
    predictions = predictions.merge(ham10000_metadata, on="image_id", how="left")
    only_lesion_predictions = only_lesion_predictions.merge(
        ham10000_metadata, on="image_id", how="left"
    )

    predictions["real_class"] = predictions.dx.map(lambda x: short_to_full_name_dict[x])
    only_lesion_predictions["real_class"] = only_lesion_predictions.dx.map(
        lambda x: short_to_full_name_dict[x]
    )

    predictions["correct"] = predictions.classification == predictions.real_class
    only_lesion_predictions["correct"] = (
        only_lesion_predictions.classification == only_lesion_predictions.real_class
    )

    # Make a confusion matrix
    confusion_matrix = (
        predictions.groupby(["real_class", "classification"]).size().unstack()
    )
    only_lesion_confusion_matrix = (
        only_lesion_predictions.groupby(["real_class", "classification"])
        .size()
        .unstack()
    )

    # Normalize over the x axis
    confusion_matrix_normalized = confusion_matrix.div(
        confusion_matrix.sum(axis=1), axis=0
    )
    only_lesion_confusion_matrix_normalized = only_lesion_confusion_matrix.div(
        only_lesion_confusion_matrix.sum(axis=1), axis=0
    )

    confusion_matrix_normalized.columns = confusion_matrix_normalized.columns.map(
        lambda x: full_name_to_short_dict[x]
    )
    only_lesion_confusion_matrix_normalized.columns = (
        only_lesion_confusion_matrix_normalized.columns.map(
            lambda x: full_name_to_short_dict[x]
        )
    )

    confusion_matrix_normalized.index = confusion_matrix_normalized.index.map(
        lambda x: full_name_to_short_dict[x]
    )
    only_lesion_confusion_matrix_normalized.index = (
        only_lesion_confusion_matrix_normalized.index.map(
            lambda x: full_name_to_short_dict[x]
        )
    )

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    p = sns.heatmap(
        confusion_matrix_normalized, annot=True, cmap="Blues", annot_kws={"size": 10}
    )
    p.set_xlabel("Predicted label")
    p.set_ylabel("True label")
    plt.savefig("only_lesion_on_original_confusion_matrix.png")

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    p = sns.heatmap(
        only_lesion_confusion_matrix_normalized,
        annot=True,
        cmap="Blues",
        annot_kws={"size": 10},
    )
    p.set_xlabel("Predicted label")
    p.set_ylabel("True label")
    plt.savefig("only_lesion_on_segmented_confusion_matrix.png")

    # Calculate precision, recall and f1 score on each of the two datasets

    # With recall we think malignant recall
    # With precision we calculate both malignant precision and general class precision
    segmented_on_normal_metrics = calculate_metrics(only_lesion_id, "normal")
    segmented_on_only_lesion_metrics = calculate_metrics(only_lesion_id, "only_lesions")
    normal_on_normal_metrics = calculate_metrics(resnet_mixup_id, "normal")
    normal_on_segmented_metrics = calculate_metrics(resnet_mixup_id, "only_lesions")

    full_image_label = "Full images"
    segmented_image_label = "Segmented"
    index = pd.MultiIndex.from_tuples(
        [
            (full_image_label, full_image_label),
            (full_image_label, segmented_image_label),
            (segmented_image_label, full_image_label),
            (segmented_image_label, segmented_image_label),
        ],
        names=["Training set", "Evaluation set"],
    )

    score_table = pd.DataFrame(
        {
            "Multi-class precision": [
                normal_on_normal_metrics["mc-precision"],
                normal_on_segmented_metrics["mc-precision"],
                segmented_on_normal_metrics["mc-precision"],
                segmented_on_only_lesion_metrics["mc-precision"],
            ],
            "Multi-class F1 score": [
                normal_on_normal_metrics["mc-f1"],
                normal_on_segmented_metrics["mc-f1"],
                segmented_on_normal_metrics["mc-f1"],
                segmented_on_only_lesion_metrics["mc-f1"],
            ],
            "Binary precision": [
                normal_on_normal_metrics["b-precision"],
                normal_on_segmented_metrics["b-precision"],
                segmented_on_normal_metrics["b-precision"],
                segmented_on_only_lesion_metrics["b-precision"],
            ],
            "Malignant F1 score": [
                normal_on_normal_metrics["b-f1"],
                normal_on_segmented_metrics["b-f1"],
                segmented_on_normal_metrics["b-f1"],
                segmented_on_only_lesion_metrics["b-f1"],
            ],
            "Malignant recall": [
                normal_on_normal_metrics["b-recall"],
                normal_on_segmented_metrics["b-recall"],
                segmented_on_normal_metrics["b-recall"],
                segmented_on_only_lesion_metrics["b-recall"],
            ],
        },
        index=index,
    )

    score_table.transpose().to_latex(
        "score_table.tex",
        bold_rows=True,
        multicolumn=True,
        # column_format="l|rr|rr|",
    )
