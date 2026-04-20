"""
Data Quality Module
Handles data quality scoring and automated insight generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def get_grade(score: float) -> str:
    """Convert score to letter grade"""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


def calculate_data_quality_score(df: pd.DataFrame, column_info: dict) -> dict:
    """
    Calculate overall data quality score (0-100).
    Based on: missing values, duplicates, variance, consistency

    Args:
        df: Input DataFrame
        column_info: Column type information from detect_column_types

    Returns:
        Dictionary with total score, breakdown, and grade
    """
    scores = {}

    # Missing values score (weight: 30)
    total_missing = sum(info['null_count'] for info in column_info.values())
    total_cells = len(df) * len(df.columns)
    missing_ratio = total_missing / total_cells if total_cells > 0 else 0
    scores['completeness'] = max(0, (1 - missing_ratio) * 100) * 0.3

    # Duplicates score (weight: 20)
    duplicate_ratio = df.duplicated().sum() / len(df)
    scores['uniqueness'] = max(0, (1 - duplicate_ratio) * 100) * 0.2

    # Variance score (weight: 25) - check for constant columns
    numeric_cols = get_columns_by_type(column_info, 'numeric')
    constant_cols = sum(1 for col in numeric_cols if df[col].nunique() <= 1)
    variance_ratio = 1 - (constant_cols / len(numeric_cols)) if numeric_cols else 1
    scores['variance'] = variance_ratio * 100 * 0.25

    # Consistency score (weight: 25) - based on data type appropriateness
    type_consistency = sum(
        1 for info in column_info.values() if not info['is_high_cardinality']
    ) / len(column_info)
    scores['consistency'] = type_consistency * 100 * 0.25

    total_score = sum(scores.values())

    return {
        'total': round(total_score, 2),
        'breakdown': {k: round(v, 2) for k, v in scores.items()},
        'grade': get_grade(total_score)
    }


def get_columns_by_type(column_info: dict, col_type: str) -> list:
    """Get list of columns of a specific type"""
    return [col for col, info in column_info.items() if info['type'] == col_type]


def generate_insights(df: pd.DataFrame, column_info: dict) -> List[dict]:
    """
    Generate automated insights about the dataset.

    Args:
        df: Input DataFrame
        column_info: Column type information

    Returns:
        List of insight dictionaries with type, title, and description
    """
    insights = []

    # Missing value insights
    high_missing = [
        (col, info['null_ratio'])
        for col, info in column_info.items()
        if info['null_ratio'] > 0.3
    ]
    if high_missing:
        for col, ratio in sorted(high_missing, key=lambda x: -x[1]):
            insights.append({
                'type': 'warning',
                'title': f'High Missing Values: {col}',
                'description': f'{ratio*100:.1f}% of values are missing in this column'
            })

    # Skewness insights for numeric columns
    numeric_cols = get_columns_by_type(column_info, 'numeric')
    for col in numeric_cols:
        if df[col].nunique() > 2:
            mean_val = df[col].mean()
            median_val = df[col].median()
            if mean_val != 0 and abs(mean_val - median_val) / abs(mean_val) > 0.3:
                skew_direction = 'right' if mean_val > median_val else 'left'
                insights.append({
                    'type': 'info',
                    'title': f'Skewed Distribution: {col}',
                    'description': f'This column shows {skew_direction} skew (mean={mean_val:.2f}, median={median_val:.2f})'
                })

    # Correlation insights
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                corr_val = corr_matrix.loc[col1, col2]
                if abs(corr_val) > 0.7:
                    high_corr.append((col1, col2, corr_val))

        for col1, col2, corr_val in high_corr:
            direction = 'positive' if corr_val > 0 else 'negative'
            insights.append({
                'type': 'success' if corr_val > 0 else 'warning',
                'title': f'Strong {direction} Correlation',
                'description': f'{col1} and {col2} are strongly correlated (r={corr_val:.2f})'
            })

    # Low variance / constant columns
    for col in numeric_cols:
        if df[col].nunique() <= 2:
            insights.append({
                'type': 'warning',
                'title': f'Low Variance: {col}',
                'description': f'This column has only {df[col].nunique()} unique values - may not be useful for analysis'
            })

    # Outlier detection using IQR
    for col in numeric_cols:
        if df[col].nunique() > 10:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            outlier_ratio = outliers / len(df)
            if outlier_ratio > 0.05:
                insights.append({
                    'type': 'warning',
                    'title': f'Potential Outliers: {col}',
                    'description': f'{outliers} rows ({outlier_ratio*100:.1f}%) contain potential outliers'
                })

    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        insights.append({
            'type': 'warning',
            'title': 'Duplicate Rows Detected',
            'description': f'{dup_count} duplicate rows found ({dup_count/len(df)*100:.1f}%)'
        })

    # Cardinality insights for categorical columns
    cat_cols = get_columns_by_type(column_info, 'categorical')
    for col in cat_cols:
        if column_info[col]['unique_count'] > 100:
            insights.append({
                'type': 'info',
                'title': f'High Cardinality: {col}',
                'description': f'This categorical column has {column_info[col]["unique_count"]} unique values'
            })

    return insights


def detect_outliers_iqr(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Detect outliers using IQR method.

    Args:
        df: Input DataFrame
        columns: List of columns to check (default: all numeric)

    Returns:
        DataFrame with boolean columns indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_df = pd.DataFrame(index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)

    outlier_df['is_outlier_any'] = outlier_df.any(axis=1)

    return outlier_df


def calculate_skewness_kurtosis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate skewness and kurtosis for all numeric columns.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with skewness and kurtosis values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = []
    for col in numeric_cols:
        stats.append({
            'column': col,
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'is_normal_skew': abs(df[col].skew()) < 0.5,
            'is_normal_kurt': abs(df[col].kurtosis()) < 0.5
        })

    return pd.DataFrame(stats)
