"""
Data Cleaning Module
Functions for data transformation, imputation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Optional, List


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop specified columns"""
    return df.drop(columns=columns, errors='ignore')


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows"""
    return df.drop_duplicates()


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[List[str]] = None,
    fill_value: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values with various strategies.

    Args:
        df: Input DataFrame
        strategy: 'mean', 'median', 'mode', 'constant', 'drop'
        columns: Columns to apply strategy to (default: all)
        fill_value: Value for 'constant' strategy

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if columns is None:
        columns = df.columns

    for col in columns:
        if col not in df.columns:
            continue

        if strategy == 'drop':
            df = df.dropna(subset=[col])
        elif strategy == 'mean' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median' and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else df[col])
        elif strategy == 'constant' and fill_value is not None:
            df[col] = df[col].fillna(fill_value)

    return df


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    new_type: str
) -> pd.DataFrame:
    """Convert column to specified type"""
    df = df.copy()
    try:
        if new_type == 'numeric':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == 'categorical':
            df[column] = df[column].astype('category')
        elif new_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == 'string':
            df[column] = df[column].astype(str)
    except Exception as e:
        raise ValueError(f"Error converting {column}: {e}")
    return df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    """Rename a column"""
    return df.rename(columns={old_name: new_name})


def create_feature(
    df: pd.DataFrame,
    new_col_name: str,
    operation: str,
    col1: str,
    col2: Optional[str] = None,
    value: Optional[str] = None
) -> pd.DataFrame:
    """Create new feature from existing columns"""
    df = df.copy()
    try:
        if operation == 'add':
            df[new_col_name] = df[col1] + df[col2]
        elif operation == 'subtract':
            df[new_col_name] = df[col1] - df[col2]
        elif operation == 'multiply':
            df[new_col_name] = df[col1] * df[col2]
        elif operation == 'divide':
            df[new_col_name] = df[col1] / df[col2].replace(0, np.nan)
        elif operation == 'log':
            df[new_col_name] = np.log1p(df[col1])
        elif operation == 'square':
            df[new_col_name] = df[col1] ** 2
        elif operation == 'sqrt':
            df[new_col_name] = np.sqrt(df[col1])
        elif operation == 'bin':
            bins = int(value) if value else 5
            df[new_col_name] = pd.cut(df[col1], bins=bins)
        elif operation == 'qcut':
            q = int(value) if value else 4
            df[new_col_name] = pd.qcut(df[col1], q=q, duplicates='drop')
        elif operation == 'ratio':
            df[new_col_name] = df[col1] / df[col2].replace(0, np.nan)
        elif operation == 'difference':
            df[new_col_name] = df[col1] - df[col2]
        elif operation == 'interaction':
            df[new_col_name] = df[col1] * df[col2]
    except Exception as e:
        raise ValueError(f"Error creating feature: {e}")
    return df


def treat_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'cap'
) -> pd.DataFrame:
    """
    Treat outliers using IQR method.

    Args:
        df: Input DataFrame
        columns: Columns to treat (default: all numeric)
        method: 'cap' (winsorize), 'remove', or 'flag'

    Returns:
        DataFrame with outliers treated
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == 'cap':
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'flag':
            df[f'{col}_outlier_flag'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

    return df


def normalize_column(
    df: pd.DataFrame,
    column: str,
    method: str = 'zscore'
) -> pd.DataFrame:
    """
    Normalize/scale a column.

    Args:
        df: Input DataFrame
        column: Column to normalize
        method: 'zscore', 'minmax', 'log', 'boxcox'

    Returns:
        DataFrame with normalized column
    """
    df = df.copy()

    if method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        df[f'{column}_normalized'] = (df[column] - mean) / std if std > 0 else 0
    elif method == 'minmax':
        min_val = df[column].min()
        max_val = df[column].max()
        df[f'{column}_normalized'] = (df[column] - min_val) / (max_val - min_val) if max_val > min_val else 0
    elif method == 'log':
        df[f'{column}_log'] = np.log1p(df[column])
    elif method == 'boxcox':
        from scipy import stats
        positive_data = df[column] - df[column].min() + 1
        transformed, _ = stats.boxcox(positive_data)
        df[f'{column}_boxcox'] = transformed

    return df


def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode categorical columns.

    Args:
        df: Input DataFrame
        columns: Columns to encode (default: all categorical)
        method: 'onehot', 'label', 'frequency', 'target'

    Returns:
        DataFrame with encoded columns
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in columns:
        if method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1).drop(columns=[col])
        elif method == 'label':
            df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        elif method == 'frequency':
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_freq'] = df[col].map(freq_map)

    return df


def bin_column(
    df: pd.DataFrame,
    column: str,
    bins: int = 5,
    labels: Optional[List[str]] = None,
    method: str = 'equal_width'
) -> pd.DataFrame:
    """
    Bin a numeric column into discrete intervals.

    Args:
        df: Input DataFrame
        column: Column to bin
        bins: Number of bins
        labels: Custom labels for bins
        method: 'equal_width' or 'equal_freq'

    Returns:
        DataFrame with binned column
    """
    df = df.copy()

    if method == 'equal_width':
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
    elif method == 'equal_freq':
        df[f'{column}_binned'] = pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')

    return df
