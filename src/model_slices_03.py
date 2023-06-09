#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this project, we used a method of model slicing where the model is trained on the entire dataset and then evaluated on individual subsets, or bins. This approach was chosen as it serves to investigate and assure the model's performance across all subsets of data, not just on average. By doing so, it provides an understanding of the model's behavior for each specific age group, which wouldn't be visible in an overall performance measure. The ultimate goal is not to assess the model's overall predictive ability, but rather to verify its adequacy for each age category. This methodology is especially useful in identifying if any age group is underrepresented or disproportionately affected by the model's performance when it is trained on data from all age groups. This ensures a more equitable and effective model application across all subsets within the data.


    The binning is done as follows:
    - '17-24: College Age / Early Career'
    - '25-34: Early Adults / Young Professionals'
    - '35-44: Midlife / Settled Adults'
    - '45-54: Late Midlife'
    - '55-64: Pre-Retirement'
    - '65-90: Retirement Age'


@author: christos
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

import pandas as pd
import os
import joblib
import logging
from typing import Optional, Any
from sklearn.metrics import roc_auc_score, fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
from model_training_02 import data_loader


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# UTILITY FUNCTIONS/CLASSES
# =============================================================================
# load the model
def load_model(
    model_path: str = "../models/trained_model.joblib",
) -> Optional[Any]:
    """
    Loads a trained machine learning model from a specified path.

    Parameters:
    model_path (str): The path (relative or absolute) to the .joblib file

    Returns:
    The trained model loaded into memory
    """
    try:
        # Load the model from the .joblib file
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"No such file or directory: '{model_path}'")
        return None
    except Exception as e:
        logging.error(f"Unexpected error occurred when loading the model: {e}")
        return None


def create_life_stage_bins(age_series: pd.Series) -> pd.Series:
    """
    Bins ages into life stages for model slicing.

    This function takes a pandas Series of ages and categorizes each age into a life stage bin.
    These life stage bins are useful for slicing models based on life stages, which can be useful
    for understanding how model performance varies across different life stages.

    The binning is done as follows:
    - '17-24: College Age / Early Career'
    - '25-34: Early Adults / Young Professionals'
    - '35-44: Midlife / Settled Adults'
    - '45-54: Late Midlife'
    - '55-64: Pre-Retirement'
    - '65-90: Retirement Age'

    Parameters:
    age_series (pd.Series): A pandas Series object containing the ages to be binned.

    Returns:
    pd.Series: A new pandas Series object of the same length as the input, where each age
               is replaced by the label of the life stage bin it falls into.
    """
    # Define bin edges and corresponding labels
    bins = [16, 24, 34, 44, 54, 64, 90]
    labels = [
        "17-24: College Age / Early Career",
        "25-34: Early Adults / Young Professionals",
        "35-44: Midlife / Settled Adults",
        "45-54: Late Midlife",
        "55-64: Pre-Retirement",
        "65-90: Retirement Age",
    ]

    # Create life stage bins
    life_stage_series = pd.cut(
        age_series, bins=bins, labels=labels, include_lowest=True
    )
    life_stage_indices = life_stage_series.groupby(life_stage_series).groups

    return life_stage_indices


# Function to plot AUC, F-beta scores, and number of samples
def plot_scores_and_samples(score_df: pd.DataFrame, n_samples: dict):
    fig, ax1 = plt.subplots()

    # Plot AUC scores
    sns.lineplot(
        data=score_df,
        x="label",
        y="AUC",
        marker="o",
        sort=False,
        ax=ax1,
        label="AUC",
        linestyle="--",
        legend=False,
    )
    ax1.set_ylabel("AUC")
    ax1.set_xlabel("age-bin")

    # Create a second y-axis
    ax2 = ax1.twinx()
    # Plot F-beta scores on the second y-axis
    sns.lineplot(
        data=score_df,
        x="label",
        y="F-beta",
        marker="o",
        sort=False,
        color="r",
        ax=ax2,
        label="F-beta",
        legend=False,
    )
    ax2.set_ylabel("F-beta")

    ax2.set_xlabel("age-bin")

    # Add the legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    # Create secondary x-axis for the number of samples
    ax3 = ax1.twiny()
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xbound(ax1.get_xbound())
    ax3.set_xticklabels(
        [n_samples[label] for label in score_df["label"]], rotation=45
    )

    # Label the secondary x-axis
    ax3.set_xlabel("Number of samples")

    plt.title("AUC and F-beta Scores per Age Group", pad=20)

    # save the figure
    path2figures = os.path.join("../figures")
    if not os.path.exists(path2figures):
        os.makedirs(path2figures)
    fname = os.path.join(path2figures, "age_binned_classification.png")
    fig.savefig(fname, bbox_inches="tight")
    plt.show()


def run_model_on_AGE_slices(raw_data):
    """
    In our investigation, we employed a particular variant of model slicing, where the model was initially trained on the comprehensive dataset and subsequently assessed on distinctive age-based subsets. This methodology was selected due to its capacity to provide nuanced insights into the model's behavior across disparate subsets of data, as opposed to a monolithic, average-oriented evaluation.

    Our objective was not to ascertain the overall predictive proficiency of the model but to determine its performance across individual age brackets. By training the model on all age groups, we could meticulously scrutinize its efficacy for each specific age cohort - an aspect typically obscured in an overarching performance measure.

    This investigative approach is instrumental in detecting potential representation bias, wherein a particular age group may be underrepresented or disproportionately impacted by the model's predictions. By ensuring the model's equitable performance across all age strata, we uphold the principles of fairness in machine learning applications, thereby contributing to the development of an effective and balanced model that can serve diverse subsets within the dataset.
    """

    # bin the AGE variable into stages
    life_stage_indices = create_life_stage_bins(raw_data.age)
    # load the model
    model_path = model_path = os.path.join(
        os.path.dirname(__file__), ".." "models", "preprocessor.joblib"
    )
    model = load_model(model_path)
    # load the preprocessed data (where the slicing will take place)
    preprocessed_data = data_loader("preprocessed")
    preprocessed_data.salary = preprocessed_data.salary.replace(-1, 0)

    # Initialize an empty list to store the score data
    scores = []

    # Loop over the slices and train the model on age slices
    for label, indices in life_stage_indices.items():
        # parse the age slice
        age_slice = preprocessed_data.iloc[indices]
        # Drop 'salary' column from age_slice as we don't want to include it in our features
        features = age_slice.drop(columns="salary")

        # Use the model to predict salary
        predictions = model.predict(features)
        # Get the true labels
        true_labels = age_slice["salary"]

        # Calculate AUC
        auc = roc_auc_score(true_labels, predictions)
        # Calculate F-beta score
        beta = 1
        fbeta = fbeta_score(true_labels, predictions, beta=beta)

        # Add scores to the list
        scores.append(
            {"label": label.split(":")[0], "AUC": auc, "F-beta": fbeta}
        )

    # Create the DataFrame
    score_df = pd.DataFrame(scores)

    # Calculate the number of samples per age group
    n_samples = {
        label.split(":")[0]: len(indices)
        for label, indices in life_stage_indices.items()
    }

    # Add this information to the score_df DataFrame
    score_df["n_samples"] = score_df["label"].map(n_samples)
    # Call the function to create the plot
    plot_scores_and_samples(score_df, n_samples)
    # store the df into the requested .txt


# =============================================================================
# MAIN CALLER
# =============================================================================


def main():
    # load the raw data
    raw_data = data_loader("raw")
    # run model on Age slices and plots the classification results
    run_model_on_AGE_slices(raw_data)


# =============================================================================
# WRAPER
# =============================================================================
if __name__ == "__main__":
    # call the main function
    main()
