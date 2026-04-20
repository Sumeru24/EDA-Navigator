"""
Data Loading Module
Handles file uploads and data loading for CSV, Excel, and JSON files.
"""

import pandas as pd
import streamlit as st
from typing import Optional


@st.cache_data
def load_csv(_file) -> Optional[pd.DataFrame]:
    """Load CSV file with optimized settings"""
    try:
        return pd.read_csv(_file, low_memory=False)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


@st.cache_data
def load_excel(_file) -> Optional[pd.DataFrame]:
    """Load Excel file"""
    try:
        return pd.read_excel(_file)
    except Exception as e:
        st.error(f"Error loading Excel: {e}")
        return None


@st.cache_data
def load_json(_file) -> Optional[pd.DataFrame]:
    """Load JSON file"""
    try:
        content = _file.read().decode('utf-8')
        return pd.read_json(content)
    except Exception as e:
        st.error(f"Error loading JSON: {e}")
        return None


def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load data from uploaded file based on file type.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        DataFrame or None if loading fails
    """
    if uploaded_file is None:
        return None

    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == 'csv':
        return load_csv(uploaded_file)
    elif file_type == 'xlsx':
        return load_excel(uploaded_file)
    elif file_type == 'json':
        return load_json(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None


def detect_column_types(
    df: pd.DataFrame,
    cardinality_threshold: float = 0.5,
    high_cardinality_threshold: int = 50
) -> dict:
    """
    Automatically detect and classify column types.

    Args:
        df: Input DataFrame
        cardinality_threshold: Ratio threshold for categorical classification
        high_cardinality_threshold: Absolute count threshold for high cardinality

    Returns:
        Dictionary with column classifications
    """
    column_info = {}

    for col in df.columns:
        col_data = df[col]
        unique_count = col_data.nunique()
        unique_ratio = unique_count / len(col_data) if len(col_data) > 0 else 0
        null_count = col_data.isnull().sum()
        null_ratio = null_count / len(col_data) if len(col_data) > 0 else 0

        # Determine type
        if pd.api.types.is_numeric_dtype(col_data):
            if unique_count <= 10 and unique_ratio < cardinality_threshold:
                col_type = 'categorical'
            else:
                col_type = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_type = 'datetime'
        else:
            # Try to parse as datetime
            try:
                parsed = pd.to_datetime(col_data, errors='coerce')
                if parsed.notna().sum() / len(col_data) > 0.8:
                    col_type = 'datetime'
                elif unique_count <= 10 or unique_ratio < cardinality_threshold:
                    col_type = 'categorical'
                else:
                    col_type = 'categorical'
            except:
                if unique_count <= 10 or unique_ratio < cardinality_threshold:
                    col_type = 'categorical'
                else:
                    col_type = 'categorical'

        is_high_cardinality = unique_count > high_cardinality_threshold

        column_info[col] = {
            'type': col_type,
            'unique_count': unique_count,
            'unique_ratio': unique_ratio,
            'null_count': null_count,
            'null_ratio': null_ratio,
            'dtype': str(col_data.dtype),
            'is_high_cardinality': is_high_cardinality
        }

    return column_info


def get_columns_by_type(column_info: dict, col_type: str) -> list:
    """Get list of columns of a specific type"""
    return [col for col, info in column_info.items() if info['type'] == col_type]
