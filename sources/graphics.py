import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_heatmap(
    dataframe: pd.DataFrame,
    name: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
) -> None:
    # Get correlation matrix
    correlation_statistics = dataframe.corr()

    # Set mask to get triangle visualization
    stst_mask = np.triu(correlation_statistics)

    # Set size for the plot
    plt.figure(figsize=(15, 15))

    # Get heatmap
    sns.heatmap(correlation_statistics, mask=stst_mask, annot=True, cmap=cmap, fmt=fmt)

    plt.title(f"Heatmap of Correlation Matrix for {name}.")


def get_count_plot(dataframe: pd.DataFrame, name: str):
    # Create a count plot for the DataFrame column
    plt.figure(figsize=(8, 6))

    sns.countplot(x=name, data=dataframe)

    plt.title(f"Count plot for {name}")

    plt.show()


def get_kde_comparison(y_data: pd.Series, x_data: pd.Series, modeler: object) -> None:
    # Get density plot
    # for test data
    sns.kdeplot(
        y_data,
        fill=False,
        color="r",
        label="test subset",
    )

    # for predicted data
    sns.kdeplot(
        modeler.predict(x_data),
        fill=True,
        color="b",
        label="predicted",
    )

    # Plot
    plt.title("Distribution of observations in test dataset and and predicted dataset")
    plt.legend()
