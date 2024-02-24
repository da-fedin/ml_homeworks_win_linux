# For typing
from typing import Dict, TypeAlias

# For Data science
import pandas as pd

from sklearn import (
    preprocessing,
    linear_model,
    metrics,
    model_selection,
)

# Set aliases for long types
LinearRegressionModel: TypeAlias = linear_model.LinearRegression
StandardScaler: TypeAlias = preprocessing.StandardScaler


def check_is_na(dataframe: pd.DataFrame) -> str:
    statement = "There is no missing values in datasets"

    if dataframe.isna().any().any():
        statement = "There are missing values in datasets"

    return statement


def get_category_encoded(
    dataset: pd.DataFrame, category_names: list[str], encoder_type: str
) -> pd.DataFrame:
    encoder = None
    encoded_df = None

    if encoder_type == "OneHotEncoder":
        encoder = preprocessing.OneHotEncoder(sparse_output=False)

        encoded_df = dataset[category_names].apply(encoder.fit_transform)
        dataset.drop(columns=category_names, inplace=True)
        result_df = pd.concat([dataset, encoded_df], axis=1)

    elif encoder_type == "LabelEncoder":
        encoder = preprocessing.LabelEncoder()

        encoded_df = dataset[category_names].apply(encoder.fit_transform)
        dataset.drop(columns=category_names, inplace=True)
        result_df = pd.concat([dataset, encoded_df], axis=1)

    else:
        print(f"No realisation for {encoder_type} has done yet")

    return result_df


def get_dataframe_scaled(
    dataset: pd.DataFrame, omit_feature_names: list[str]
) -> pd.DataFrame:
    # Set columns to scale
    columns_to_scale = dataset.columns.difference(omit_feature_names)
    data_to_scale = dataset[columns_to_scale]

    # Get scaler
    scaler = preprocessing.StandardScaler().fit(data_to_scale)

    # Get scaled data
    scaled_data = scaler.transform(data_to_scale)

    # Create a new DataFrame with the scaled data
    scaled_data_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
    scaled_data_df[omit_feature_names] = dataset[omit_feature_names]

    return scaled_data_df


def evaluate_model(
    model: LinearRegressionModel,
    x_train: StandardScaler,
    x_test: StandardScaler,
    y_train: StandardScaler,
    y_test: StandardScaler,
) -> Dict[str, float]:
    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)

    # Get mean square metrics
    mse_train = metrics.mean_squared_error(y_train, y_train_predicted)
    mse_test = metrics.mean_squared_error(y_test, y_test_predicted)

    # Get RMS metrics
    rmse_train = metrics.mean_squared_error(y_train, y_train_predicted, squared=False)
    rmse_test = metrics.mean_squared_error(y_test, y_test_predicted, squared=False)

    return {
        "MSE trained": round(mse_train, 3),
        "MSE tested": round(mse_test, 3),
        "RMSE trained": round(rmse_train, 3),
        "RMSE tested": round(rmse_test, 3),
    }


def get_model_score(y_data: pd.Series, x_data: pd.Series, modeler: object) -> None:
    # Get cross-validation results
    cross_validation_score = model_selection.cross_val_score(
        estimator=modeler, X=x_data, y=y_data, cv=3
    )

    # Get fit
    modeler.fit(X=x_data, y=y_data)

    # Get train score
    train_score = modeler.score(X=x_data, y=y_data)
    # Geet test score
    test_score = modeler.score(X=x_data, y=y_data)

    print(f"Cross validation score: {cross_validation_score[:]}")
    print(f"Train score: {train_score:.2f}")
    print(f"Test score: {test_score:.2f}")


def three_sigma_cleared(
    dataset: pd.DataFrame, feature_names: list[str], sigmas=3
) -> pd.DataFrame:
    dataset_filtered = None

    for column_name in feature_names:
        data_column = dataset[column_name]

        # Calculate mean and standard deviation
        mean_value = data_column.mean()
        std_value = data_column.std()

        # Define the threshold range
        lower_threshold = mean_value - sigmas * std_value
        upper_threshold = mean_value + sigmas * std_value

        # Identify values outside the range
        outliers_mask = (data_column < lower_threshold) | (
            data_column > upper_threshold
        )

        # Exclude or replace outliers as needed
        dataset_filtered = dataset[~outliers_mask]

    return dataset_filtered
