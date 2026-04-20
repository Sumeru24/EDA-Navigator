"""
Statistical Analysis Module
Hypothesis testing, normality tests, and advanced statistical functions.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from itertools import combinations


def test_normality(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Test normality of distributions using Shapiro-Wilk test.

    Args:
        df: Input DataFrame
        columns: Columns to test (default: all numeric)

    Returns:
        DataFrame with test results
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col in columns:
        data = df[col].dropna()
        if len(data) < 3:
            continue

        stat, p_value = stats.shapiro(data)
        is_normal = p_value > 0.05

        results.append({
            'column': col,
            'shapiro_stat': stat,
            'p_value': p_value,
            'is_normal': is_normal,
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        })

    return pd.DataFrame(results)


def test_correlation_significance(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = 'pearson'
) -> Dict:
    """
    Test correlation significance between two columns.

    Args:
        df: Input DataFrame
        col1: First column
        col2: Second column
        method: 'pearson', 'spearman', or 'kendall'

    Returns:
        Dictionary with correlation results
    """
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()

    # Align indices
    common_idx = data1.index.intersection(data2.index)
    data1 = data1.loc[common_idx]
    data2 = data2.loc[common_idx]

    corr, p_value = stats.pearsonr(data1, data2) if method == 'pearson' else \
                    stats.spearmanr(data1, data2) if method == 'spearman' else \
                    stats.kendalltau(data1, data2)

    return {
        'col1': col1,
        'col2': col2,
        'method': method,
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(data1)
    }


def anova_test(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> Dict:
    """
    Perform one-way ANOVA test.

    Args:
        df: Input DataFrame
        numeric_col: Numeric dependent variable
        group_col: Categorical grouping variable

    Returns:
        Dictionary with ANOVA results
    """
    groups = df[group_col].unique()
    samples = [df[df[group_col] == g][numeric_col].dropna().values for g in groups]

    f_stat, p_value = stats.f_oneway(*samples)

    return {
        'numeric_col': numeric_col,
        'group_col': group_col,
        'n_groups': len(groups),
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def chi_square_test(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Dict:
    """
    Perform chi-square test of independence.

    Args:
        df: Input DataFrame
        col1: First categorical column
        col2: Second categorical column

    Returns:
        Dictionary with chi-square test results
    """
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return {
        'col1': col1,
        'col2': col2,
        'chi_square': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'significant': p_value < 0.05,
        'contingency_table': contingency,
        'expected_frequencies': expected
    }


def t_test_independent(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1: str,
    group2: str
) -> Dict:
    """
    Perform independent samples t-test.

    Args:
        df: Input DataFrame
        numeric_col: Numeric dependent variable
        group_col: Categorical grouping variable
        group1: First group name
        group2: Second group name

    Returns:
        Dictionary with t-test results
    """
    data1 = df[df[group_col] == group1][numeric_col].dropna()
    data2 = df[df[group_col] == group2][numeric_col].dropna()

    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Welch's t-test

    return {
        'numeric_col': numeric_col,
        'group_col': group_col,
        'group1': group1,
        'group2': group2,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': data1.mean() - data2.mean()
    }


def t_test_paired(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Dict:
    """
    Perform paired samples t-test.

    Args:
        df: Input DataFrame
        col1: First measurement column
        col2: Second measurement column

    Returns:
        Dictionary with paired t-test results
    """
    data = df[[col1, col2]].dropna()

    t_stat, p_value = stats.ttest_rel(data[col1], data[col2])

    return {
        'col1': col1,
        'col2': col2,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'mean_diff': data[col1].mean() - data[col2].mean(),
        'n_pairs': len(data)
    }


def mann_whitney_u(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1: str,
    group2: str
) -> Dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        df: Input DataFrame
        numeric_col: Numeric dependent variable
        group_col: Categorical grouping variable
        group1: First group name
        group2: Second group name

    Returns:
        Dictionary with test results
    """
    data1 = df[df[group_col] == group1][numeric_col].dropna()
    data2 = df[df[group_col] == group2][numeric_col].dropna()

    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    return {
        'numeric_col': numeric_col,
        'group_col': group_col,
        'group1': group1,
        'group2': group2,
        'u_statistic': u_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def kruskal_wallis(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str
) -> Dict:
    """
    Perform Kruskal-Wallis H test (non-parametric alternative to ANOVA).

    Args:
        df: Input DataFrame
        numeric_col: Numeric dependent variable
        group_col: Categorical grouping variable

    Returns:
        Dictionary with test results
    """
    groups = df[group_col].unique()
    samples = [df[df[group_col] == g][numeric_col].dropna().values for g in groups]

    h_stat, p_value = stats.kruskal(*samples)

    return {
        'numeric_col': numeric_col,
        'group_col': group_col,
        'n_groups': len(groups),
        'h_statistic': h_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def run_all_correlation_tests(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Run correlation tests for all pairs of numeric columns.

    Args:
        df: Input DataFrame
        numeric_cols: Columns to include (default: all numeric)
        method: Correlation method

    Returns:
        DataFrame with all pairwise correlations
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    results = []
    for col1, col2 in combinations(numeric_cols, 2):
        result = test_correlation_significance(df, col1, col2, method)
        results.append(result)

    return pd.DataFrame(results)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive descriptive statistics.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with descriptive statistics
    """
    stats_list = []

    for col in df.columns:
        data = df[col].dropna()

        if pd.api.types.is_numeric_dtype(data):
            stats_dict = {
                'column': col,
                'dtype': str(data.dtype),
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'q1': data.quantile(0.25),
                'median': data.median(),
                'q3': data.quantile(0.75),
                'max': data.max(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                'variance': data.var(),
                'cv': data.std() / data.mean() if data.mean() != 0 else np.nan
            }
        else:
            stats_dict = {
                'column': col,
                'dtype': str(data.dtype),
                'count': len(data),
                'unique': data.nunique(),
                'mode': str(data.mode().iloc[0]) if len(data.mode()) > 0 else None,
                'mode_freq': data.value_counts().iloc[0] if len(data.value_counts()) > 0 else 0
            }

        stats_list.append(stats_dict)

    return pd.DataFrame(stats_list)


def calculate_effect_size(
    df: pd.DataFrame,
    numeric_col: str,
    group_col: str,
    group1: str,
    group2: str,
    effect_type: str = 'cohens_d'
) -> Dict:
    """
    Calculate effect size for group comparisons.

    Args:
        df: Input DataFrame
        numeric_col: Numeric dependent variable
        group_col: Categorical grouping variable
        group1: First group
        group2: Second group
        effect_type: 'cohens_d', 'hedges_g', or 'eta_squared'

    Returns:
        Dictionary with effect size
    """
    data1 = df[df[group_col] == group1][numeric_col].dropna()
    data2 = df[df[group_col] == group2][numeric_col].dropna()

    n1, n2 = len(data1), len(data2)
    mean1, mean2 = data1.mean(), data2.mean()
    std1, std2 = data1.std(), data2.std()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    if effect_type == 'cohens_d':
        effect_size = (mean1 - mean2) / pooled_std
        interpretation = 'negligible' if abs(effect_size) < 0.2 else \
                        'small' if abs(effect_size) < 0.5 else \
                        'medium' if abs(effect_size) < 0.8 else 'large'
    elif effect_type == 'hedges_g':
        d = (mean1 - mean2) / pooled_std
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        effect_size = d * correction
        interpretation = 'negligible' if abs(effect_size) < 0.2 else \
                        'small' if abs(effect_size) < 0.5 else \
                        'medium' if abs(effect_size) < 0.8 else 'large'
    else:
        effect_size = None
        interpretation = 'N/A'

    return {
        'effect_size': effect_size,
        'interpretation': interpretation,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'pooled_std': pooled_std
    }


def pca_analysis(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95
) -> Dict:
    """
    Perform Principal Component Analysis.

    Args:
        df: Input DataFrame
        columns: Columns to include (default: all numeric)
        n_components: Number of components (default: determined by variance_threshold)
        variance_threshold: Variance to retain

    Returns:
        Dictionary with PCA results
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[columns].dropna()

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Determine n_components if not specified
    if n_components is None:
        pca_temp = PCA(random_state=42)
        pca_temp.fit(data_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1

    # Final PCA
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(data_scaled)

    # Create DataFrame with principal components
    pc_df = pd.DataFrame(
        components,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.index
    )

    # Variance explained
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Variance Explained': pca.explained_variance_ratio_,
        'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
    })

    # Loadings
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=columns
    )

    return {
        'components_df': pc_df,
        'variance_df': variance_df,
        'loadings_df': loadings_df,
        'n_components': n_components,
        'total_variance_explained': pca.explained_variance_ratio_.sum()
    }
