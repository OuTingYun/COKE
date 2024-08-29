import argparse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import sklearn.neighbors._base
import sys
import warnings

warnings.filterwarnings("ignore")

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from missingpy import MissForest


def get_missing_rate(data: pd.DataFrame | pd.Series) -> float:
    if isinstance(data, pd.Series):
        return data.isna().sum() / len(data)
    else:  # it's a DataFrame
        return data.isna().sum().sum() / (data.shape[0] * data.shape[1])


def mean_imputation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    strategy = kwargs.pop("strategy", "mean")
    imputer = SimpleImputer(strategy=strategy, **kwargs)
    imputed_df = imputer.fit_transform(df)
    return pd.DataFrame(imputed_df, columns=df.columns, index=df.index)


def missforest_imputation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # MissForest doesn't work with only one column
    if len(df.columns) == 1:
        return df

    imputer = MissForest(**kwargs)
    imputed_df = imputer.fit_transform(df)
    return pd.DataFrame(imputed_df, columns=df.columns, index=df.index)


def knn_imputation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    imputer = KNNImputer(**kwargs)
    imputed_df = imputer.fit_transform(df)
    return pd.DataFrame(imputed_df, columns=df.columns, index=df.index)


def mice_imputation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    imputer = IterativeImputer(**kwargs)
    imputed_df = imputer.fit_transform(df)
    return pd.DataFrame(imputed_df, columns=df.columns, index=df.index)


def apply_imputation(df: pd.DataFrame, imp_method: str, **kwargs) -> pd.DataFrame:
    # Identify all-NaN columns
    nan_cols = df.columns[df.isnull().all()].tolist()
    if len(nan_cols) == len(df.columns):
        return df  # All columns are NaN, so we can't impute anything

    # Remove the all-NaN columns temporarily for imputation
    df_to_impute = df.drop(nan_cols, axis=1)

    # Apply imputation
    if imp_method == "mean":
        imputed_df = mean_imputation(df_to_impute, **kwargs)
    elif imp_method == "missforest":
        imputed_df = missforest_imputation(df_to_impute, **kwargs)
    elif imp_method == "knn":
        imputed_df = knn_imputation(df_to_impute, **kwargs)
    elif imp_method == "mice":
        imputed_df = mice_imputation(df_to_impute, **kwargs)
    elif imp_method == "drop_row":
        imputed_df = df_to_impute.dropna(axis=0)
    else:
        raise ValueError(f"Invalid imputation method: {imp_method}")

    # Re-add all-NaN columns after imputation
    for col in nan_cols:
        imputed_df[col] = np.nan

    # Reorder columns to match the original order
    imputed_df = imputed_df.reindex(columns=df.columns)
    assert imputed_df.columns.tolist() == df.columns.tolist()

    return imputed_df


def data_imputation(
    df: pd.DataFrame,  # only feature columns
    imp_method: str,
    **kwargs,
) -> pd.DataFrame:
    original_df = df.copy()  # just for checking the shape of the imputed df


    # Impute globally if we still have missing values
    if get_missing_rate(df) > 0:
        df = apply_imputation(df, imp_method, **kwargs)

    # Check the shape of the imputed df
    if imp_method == "drop_row":
        assert len(df) > 0, "Dropped all rows by drop_row"
        print(f"Dropped {len(original_df) - len(df)} rows, {len(df)} rows remaini ng")
    else:
        assert df.shape == df.shape

    # Confirm that no missing values remain
    assert get_missing_rate(df) == 0, "Imputation failed"

    return df



parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="data")
parser.add_argument("--mr", type=float, default=0.52)
parser.add_argument("--rep", type=int, default=10)
parser.add_argument("--n_full", type=int, default=100)
parser.add_argument("--imp_method", type=str, default="missforest")

args = parser.parse_args()


f = args.folder
mr = args.mr
rep = args.rep
xfull = args.n_full
path = f'./{f}/X_miss({mr})_rep({rep})_xfull({xfull}).csv'
print("start:",path)
df = pd.read_csv(path)
imp_params = {}
impute_df = data_imputation(df, args.imp_method, **imp_params)
impute_df.to_csv(f'./{f}/X_miss({mr})_rep({rep})_xfull({xfull})_imp({args.imp_method}).csv', index=False)
print("finish impute:",args.imp_method)