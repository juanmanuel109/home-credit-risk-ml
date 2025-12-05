from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):

    # identify categorical columns
    categorical_cols = working_train_df.select_dtypes(
        include=["object"]
    ).columns.to_list()

    # separate columns with 2 categories and more than 2 categories
    binary_cols = []
    multi_cols = []

    for col in categorical_cols:
        n_categories = working_train_df[col].nunique()
        if n_categories == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    # encode binary categorical columns with OrdinalEncoder
    if binary_cols:
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        ordinal_encoder.fit(working_train_df[binary_cols])

        working_train_df[binary_cols] = ordinal_encoder.transform(
            working_train_df[binary_cols]
        )
        working_val_df[binary_cols] = ordinal_encoder.transform(
            working_val_df[binary_cols]
        )
        working_test_df[binary_cols] = ordinal_encoder.transform(
            working_test_df[binary_cols]
        )

    # encode multi-category categorical columns with OneHotEncoder
    if multi_cols:
        onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        onehot_encoder.fit(working_train_df[multi_cols])

        # transform and obtain the new column names
        train_encoded = onehot_encoder.transform(working_train_df[multi_cols])
        val_encoded = onehot_encoder.transform(working_val_df[multi_cols])
        test_encoded = onehot_encoder.transform(working_test_df[multi_cols])

        # obtain new column names encoding
        encoded_feature_names = onehot_encoder.get_feature_names_out(multi_cols)

        # create new dataframes with encoded features
        train_encoded_df = pd.DataFrame(
            train_encoded, columns=encoded_feature_names, index=working_train_df.index
        )
        val_encoded_df = pd.DataFrame(
            val_encoded, columns=encoded_feature_names, index=working_val_df.index
        )
        test_encoded_df = pd.DataFrame(
            test_encoded, columns=encoded_feature_names, index=working_test_df.index
        )

        # drop original multi-category columns and concatenate the new encoded columns
        working_train_df = pd.concat(
            [working_train_df.drop(columns=multi_cols), train_encoded_df], axis=1
        )
        working_val_df = pd.concat(
            [working_val_df.drop(columns=multi_cols), val_encoded_df], axis=1
        )
        working_test_df = pd.concat(
            [working_test_df.drop(columns=multi_cols), test_encoded_df], axis=1
        )

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    imputer = SimpleImputer(strategy="median")
    imputer.fit(working_train_df)

    working_train_df = pd.DataFrame(
        imputer.transform(working_train_df),
        columns=working_train_df.columns,
        index=working_train_df.index,
    )
    working_val_df = pd.DataFrame(
        imputer.transform(working_val_df),
        columns=working_val_df.columns,
        index=working_val_df.index,
    )
    working_test_df = pd.DataFrame(
        imputer.transform(working_test_df),
        columns=working_test_df.columns,
        index=working_test_df.index,
    )

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    scaler = MinMaxScaler()
    scaler.fit(working_train_df)

    train = scaler.transform(working_train_df)
    val = scaler.transform(working_val_df)
    test = scaler.transform(working_test_df)
    return train, val, test
