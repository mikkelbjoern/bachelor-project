import pandas as pd
from src.utils import DATA_FOLDER, HAM10000_DATA_FOLDER, scientific_notation
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def build_confounder_label_correlation():
    print("Building confounder label correlation...")

    # Load the data
    confounders = pd.read_csv(DATA_FOLDER + "/confounder_labels.csv")
    labels = pd.read_csv(HAM10000_DATA_FOLDER + "/HAM10000_metadata.csv")
    # Renaming to match the two datasets
    confounders = confounders.rename(columns={"image": "image_id"})

    # Merge the two datasets
    merged = pd.merge(confounders, labels, on="image_id")
    
    # Make a confusion matrix
    confusion_matrix = merged.groupby(["ruler", "dx"]).size().unstack()

    # Normalize over the x axis
    confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 4))
    sns.heatmap(confusion_matrix_normalized, annot=True, cmap="Blues")
    plt.savefig("confusion_matrix_seaborn.png")
    plt.clf()

    # Perform chi-square test on the original confusion matrix
    chi2, p, dof, expected = stats.chi2_contingency(confusion_matrix)
    
    with open("chi2.txt", "w") as f:
        f.write(str(round(chi2, 2)) + "\n")

    with open("p.txt", "w") as f:
        f.write(scientific_notation(p) + "\n")

    with open("dof.txt", "w") as f:
        f.write(str(dof) + "\n")