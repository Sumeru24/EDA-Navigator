"""
DataInsight Pro - Production-Grade EDA & ML Platform
Refactored modular architecture with advanced features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from datetime import datetime

# Import from modular packages
from src.data_loader import load_data, detect_column_types, get_columns_by_type
from src.data_quality import calculate_data_quality_score, generate_insights, detect_outliers_iqr
from src.visualizations import (
    create_histogram, create_boxplot, create_scatter_plot, create_line_chart,
    create_bar_chart, create_correlation_heatmap, create_pair_plot, create_facet_plot,
    create_3d_scatter, create_violin_plot, create_treemap, create_sunburst,
    create_animated_scatter, create_waterfall, create_parallel_categories,
    create_density_heatmap, create_radar_chart
)
from src.cleaning import (
    drop_columns, drop_duplicates, handle_missing_values, convert_column_type,
    rename_column, create_feature, treat_outliers_iqr, normalize_column,
    encode_categorical, bin_column
)
from src.ml_pipeline import (
    determine_problem_type, prepare_data, train_classification_model,
    train_regression_model, cross_validate, hyperparameter_tuning,
    train_ensemble_model, compare_models, get_feature_importance
)
from src.statistics import (
    test_normality, test_correlation_significance, anova_test, chi_square_test,
    t_test_independent, descriptive_statistics, pca_analysis, run_all_correlation_tests,
    mann_whitney_u, kruskal_wallis
)
from src.advanced_filters import QueryBuilder, FilterPresets, ConditionalFormatting, create_smart_filter
from src.reporting import generate_summary_report, create_export_package
from src.ui_components import (
    render_header, render_metric_cards, render_data_quality_breakdown,
    render_insights, render_sidebar, create_download_buttons,
    render_cross_validation_results, render_hyperparameter_results,
    render_feature_importance, render_confusion_matrix, render_statistical_test_results,
    render_pca_results, initialize_premium_ui, render_section_header
)

warnings.filterwarnings('ignore')

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'df': None,
        'df_original': None,
        'cleaned': False,
        'filters_applied': False,
        'target_column': None,
        'column_types': {},
        'theme': 'light',
        'current_section': 'Upload',
        'insights': [],
        'model_results': None,
        'data_quality_score': None,
        'query_builder': None,
        'filter_presets': FilterPresets(),
        'saved_dashboards': {},
        'statistical_results': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DataInsight Pro - EDA & ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': "DataInsight Pro - Professional EDA & ML Platform"
    }
)

# ============================================================================
# PREMIUM UI INITIALIZATION
# ============================================================================

initialize_premium_ui()

# ============================================================================
# PAGE RENDERERS
# ============================================================================

def render_upload_page():
    """Render file upload page"""
    render_header("📤 Upload Your Data", "Upload a CSV, Excel, or JSON file to begin analysis")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'json'],
        help="Supported formats: CSV, Excel (.xlsx), JSON"
    )

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)

            if df is not None:
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.session_state.column_types = detect_column_types(df)
                st.session_state.data_quality_score = calculate_data_quality_score(df, st.session_state.column_types)
                st.session_state.insights = generate_insights(df, st.session_state.column_types)
                st.session_state.cleaned = False
                st.session_state.query_builder = QueryBuilder(df)

                st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns!")

                # File metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                with col4:
                    st.metric("Quality Score", f"{st.session_state.data_quality_score['total']}/100")

                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                st.markdown("### Detected Column Types")
                type_df = pd.DataFrame([
                    {'Column': col, 'Type': info['type'], 'Dtype': info['dtype'],
                     'Unique': info['unique_count'], 'Missing': f"{info['null_ratio']*100:.1f}%"}
                    for col, info in st.session_state.column_types.items()
                ])
                st.dataframe(type_df, use_container_width=True)


def render_overview_page():
    """Render data overview page"""
    df = st.session_state.df
    column_info = st.session_state.column_types

    render_header("📊 Data Overview", "Comprehensive dataset summary and statistics")

    tabs = st.tabs(["📋 Summary", "📊 Head", "📉 Tail", "🎲 Sample", "📝 Types", "📈 Statistics"])

    with tabs[0]:
        render_metric_cards({
            'Total Rows': f"{len(df):,}",
            'Total Columns': len(df.columns),
            'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'Duplicate Rows': df.duplicated().sum(),
            'Quality Score': f"{st.session_state.data_quality_score['total']}/100"
        }, cols=5)

        render_data_quality_breakdown(st.session_state.data_quality_score)

        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("### Missing Values Summary")
        missing_df = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(missing_df, use_container_width=True)

    with tabs[1]:
        st.dataframe(df.head(10), use_container_width=True)

    with tabs[2]:
        st.dataframe(df.tail(10), use_container_width=True)

    with tabs[3]:
        sample_size = st.slider("Sample Size", min_value=5, max_value=min(100, len(df)), value=10)
        st.dataframe(df.sample(sample_size, random_state=42), use_container_width=True)

    with tabs[4]:
        type_df = pd.DataFrame([
            {'Column': col, 'Type': info['type'], 'Dtype': info['dtype'],
             'Unique': info['unique_count'], 'Missing %': f"{info['null_ratio']*100:.1f}"}
            for col, info in column_info.items()
        ])
        st.dataframe(type_df, use_container_width=True)

        st.markdown("### Override Column Types")
        col1, col2, col3 = st.columns(3)
        with col1:
            col_to_change = st.selectbox("Select Column", df.columns.tolist(), key="override_col")
        with col2:
            new_type = st.selectbox("New Type", ['numeric', 'categorical', 'datetime', 'string'], key="override_type")
        with col3:
            if st.button("Update", use_container_width=True, key="override_btn"):
                st.session_state.df = convert_column_type(st.session_state.df, col_to_change, new_type)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success(f"Updated {col_to_change} to {new_type}")
                st.rerun()

    with tabs[5]:
        st.markdown("### Comprehensive Descriptive Statistics")
        stats_df = descriptive_statistics(df)
        st.dataframe(stats_df, use_container_width=True)


def render_cleaning_page():
    """Render data cleaning page"""
    df = st.session_state.df

    render_header("🧹 Data Cleaning", "Transform and clean your data")

    tabs = st.tabs(["🗑️ Drop", "❓ Missing", "🔧 Transform", "➕ Features", "🎯 Outliers"])

    with tabs[0]:
        st.markdown("### Drop Columns and Duplicates")

        col1, col2 = st.columns(2)
        with col1:
            cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
            if st.button("Drop Selected Columns", use_container_width=True):
                st.session_state.df = drop_columns(df, cols_to_drop)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success(f"Dropped {len(cols_to_drop)} columns")
                st.rerun()

        with col2:
            dup_count = df.duplicated().sum()
            st.info(f"Duplicate rows: {dup_count}")
            if st.button("Remove Duplicates", use_container_width=True):
                st.session_state.df = drop_duplicates(df)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success(f"Removed {dup_count} duplicates")
                st.rerun()

    with tabs[1]:
        st.markdown("### Handle Missing Values")

        missing_summary = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Missing': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).values
        })
        missing_summary = missing_summary[missing_summary['Missing'] > 0]

        if len(missing_summary) > 0:
            st.dataframe(missing_summary, use_container_width=True)

            strategy = st.selectbox("Strategy", ['mean', 'median', 'mode', 'constant', 'drop'])
            cols_to_impute = st.multiselect("Select columns", missing_summary['Column'].tolist())

            fill_value = None
            if strategy == 'constant':
                fill_value = st.text_input("Fill Value", "0")

            if st.button("Apply Imputation"):
                st.session_state.df = handle_missing_values(df, strategy, cols_to_impute, fill_value)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success("Missing values handled")
                st.rerun()
        else:
            st.success("No missing values!")

    with tabs[2]:
        st.markdown("### Transform Data")

        col1, col2 = st.columns(2)
        with col1:
            col_to_rename = st.selectbox("Rename Column", df.columns.tolist(), key="rename_col")
            new_name = st.text_input("New Name", key="new_name")
        with col2:
            col_to_convert = st.selectbox("Convert Column", df.columns.tolist(), key="convert_col")
            target_type = st.selectbox("To Type", ['numeric', 'categorical', 'datetime', 'string'], key="target_type")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rename", use_container_width=True):
                if new_name and new_name != col_to_rename:
                    st.session_state.df = rename_column(df, col_to_rename, new_name)
                    st.session_state.column_types = detect_column_types(st.session_state.df)
                    st.success(f"Renamed to {new_name}")
                    st.rerun()
        with c2:
            if st.button("Convert", use_container_width=True):
                st.session_state.df = convert_column_type(df, col_to_convert, target_type)
                st.session_state.column_types = detect_column_types(st.session_state.df)
                st.success(f"Converted to {target_type}")
                st.rerun()

        st.markdown("### Normalization")
        norm_col = st.selectbox("Column to normalize", df.select_dtypes(include=[np.number]).columns.tolist())
        norm_method = st.selectbox("Method", ['zscore', 'minmax', 'log'])
        if st.button("Normalize Column"):
            st.session_state.df = normalize_column(df, norm_col, norm_method)
            st.success(f"Normalized {norm_col} using {norm_method}")
            st.rerun()

    with tabs[3]:
        st.markdown("### Create New Feature")

        col1, col2 = st.columns(2)
        with col1:
            new_col_name = st.text_input("New Column Name")
            operation = st.selectbox("Operation", [
                'add', 'subtract', 'multiply', 'divide', 'ratio',
                'log', 'square', 'sqrt', 'bin', 'qcut'
            ])
        with col2:
            col1_op = st.selectbox("First Column", df.columns.tolist(), key="feat1")
            if operation not in ['log', 'square', 'sqrt']:
                col2_op = st.selectbox("Second Column", df.columns.tolist(), key="feat2")
            else:
                col2_op = None
            value = st.text_input("Bins/Quantiles", "5") if operation in ['bin', 'qcut'] else None

        if st.button("Create Feature"):
            st.session_state.df = create_feature(df, new_col_name, operation, col1_op, col2_op, value)
            st.session_state.column_types = detect_column_types(st.session_state.df)
            st.success(f"Created {new_col_name}")
            st.rerun()

    with tabs[4]:
        st.markdown("### Outlier Treatment")

        outlier_cols = st.multiselect(
            "Select columns for outlier treatment",
            df.select_dtypes(include=[np.number]).columns.tolist()
        )
        method = st.selectbox("Treatment Method", ['cap', 'remove', 'flag'])

        if st.button("Treat Outliers"):
            st.session_state.df = treat_outliers_iqr(df, outlier_cols, method)
            st.session_state.column_types = detect_column_types(st.session_state.df)
            st.success(f"Outliers treated using {method} method")
            st.rerun()

        # Show outlier detection preview
        if outlier_cols:
            st.markdown("### Outlier Preview")
            outlier_df = detect_outliers_iqr(df, outlier_cols)
            st.dataframe(outlier_df.head(10), use_container_width=True)


def render_filters_page():
    """Render advanced filtering page"""
    df = st.session_state.df

    render_header("🔍 Advanced Filters", "Query builder and filter presets")

    # Initialize query builder
    if st.session_state.query_builder is None:
        st.session_state.query_builder = QueryBuilder(df)

    tabs = st.tabs(["🛠️ Query Builder", "💾 Presets", "🎨 Conditional Formatting"])

    with tabs[0]:
        st.markdown("### Build Custom Query")

        # Add condition
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            filter_col = st.selectbox("Column", df.columns.tolist(), key="filter_col_select")
        with col2:
            operators = ['==', '!=', '>', '<', '>=', '<=', 'in', 'not in',
                        'contains', 'is null', 'is not null']
            operator = st.selectbox("Operator", operators)
        with col3:
            if operator in ['is null', 'is not null']:
                value = None
            elif operator in ['in', 'not in']:
                value = st.text_input("Values (comma-separated)", key="filter_value")
            else:
                value = st.text_input("Value", key="filter_value_simple")
        with col4:
            if st.button("Add Condition", use_container_width=True):
                if operator in ['in', 'not in'] and value:
                    value = [v.strip() for v in value.split(',')]
                st.session_state.query_builder.add_condition(filter_col, operator, value)
                st.success("Condition added!")

        st.markdown(f"### Active Conditions ({st.session_state.query_builder.get_condition_count()})")
        for i, cond in enumerate(st.session_state.query_builder.conditions):
            st.write(f"{i+1}. {cond[0]} {cond[1]} {cond[2]}")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Apply (AND)", use_container_width=True):
                st.session_state.df = st.session_state.query_builder.apply_and()
                st.session_state.filters_applied = True
                st.success(f"Filtered to {len(st.session_state.df):,} rows")
                st.rerun()
        with c2:
            if st.button("Apply (OR)", use_container_width=True):
                st.session_state.df = st.session_state.query_builder.apply_or()
                st.session_state.filters_applied = True
                st.success(f"Filtered to {len(st.session_state.df):,} rows")
                st.rerun()
        with c3:
            if st.button("Clear All", use_container_width=True):
                st.session_state.query_builder.clear()
                st.session_state.df = st.session_state.df_original.copy()
                st.session_state.filters_applied = False
                st.success("Filters cleared")
                st.rerun()

        # Smart filter suggestions
        st.markdown("### Smart Filter Suggestions")
        smart_col = st.selectbox("Select column for suggestions", df.columns.tolist(), key="smart_col")
        suggestions = create_smart_filter(df, smart_col)
        for key, sug in suggestions.items():
            st.write(f"- **{sug.get('description', key)}**: {sug['type']}")

    with tabs[1]:
        st.markdown("### Filter Presets")

        preset_name = st.text_input("Preset Name")

        if st.button("Save Current Filters"):
            if preset_name:
                st.session_state.filter_presets.save_preset(
                    preset_name,
                    st.session_state.query_builder.conditions.copy()
                )
                st.success(f"Saved preset '{preset_name}'")

        st.markdown("### Saved Presets")
        presets = st.session_state.filter_presets.list_presets()
        if presets:
            for preset in presets:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{preset}**")
                with col2:
                    if st.button("Load", key=f"load_{preset}"):
                        st.session_state.df = st.session_state.filter_presets.apply_preset(preset, st.session_state.df_original)
                        st.success(f"Loaded '{preset}'")
                        st.rerun()
                    if st.button("Delete", key=f"del_{preset}"):
                        st.session_state.filter_presets.delete_preset(preset)
                        st.success(f"Deleted '{preset}'")
                        st.rerun()

    with tabs[2]:
        st.markdown("### Conditional Formatting")

        fmt_col = st.selectbox("Column to format", df.select_dtypes(include=[np.number]).columns.tolist())
        fmt_type = st.selectbox("Format Type", ['highlight_max', 'highlight_min', 'highlight_outliers', 'gradient'])
        color = st.color_picker("Highlight Color", "#d4edda")

        if st.button("Apply Formatting"):
            formatter = ConditionalFormatting(df)
            if fmt_type == 'highlight_max':
                formatter.add_highlight_max(fmt_col, color)
            elif fmt_type == 'highlight_min':
                formatter.add_highlight_min(fmt_col, color)
            elif fmt_type == 'highlight_outliers':
                formatter.add_highlight_outliers(fmt_col, color=color)
            elif fmt_type == 'gradient':
                formatter.add_gradient(fmt_col)

            styled = formatter.apply()
            st.dataframe(styled, use_container_width=True)


def render_visualizations_page():
    """Render visualizations page with advanced charts"""
    df = st.session_state.df
    column_info = st.session_state.column_types

    render_header("📈 Visualizations", "Standard and advanced visualization options")

    numeric_cols = get_columns_by_type(column_info, 'numeric')
    categorical_cols = get_columns_by_type(column_info, 'categorical')
    datetime_cols = get_columns_by_type(column_info, 'datetime')

    tabs = st.tabs([
        "📊 Distribution", "📦 Comparison", "🔵 Relationship", "📈 Time Series",
        "🔥 Correlation", "✨ Advanced", "🎨 Custom"
    ])

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            hist_col = st.selectbox("Column", df.columns.tolist(), key="hist_col")
            bins = st.slider("Bins", 10, 100, 30)
        with col2:
            show_kde = st.checkbox("Show KDE", value=True)
            if st.button("Generate Histogram"):
                fig = create_histogram(df, hist_col, bins=bins, kde=show_kde)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Violin Plot")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            v_x = st.selectbox("Category", categorical_cols if categorical_cols else df.columns.tolist(), key="viol_x")
        with v_col2:
            v_y = st.selectbox("Value", numeric_cols, key="viol_y")
        if st.button("Generate Violin Plot"):
            fig = create_violin_plot(df, v_x, v_y)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### Box Plot")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            b_y = st.selectbox("Y Column", numeric_cols, key="box_y")
        with b_col2:
            b_color = st.selectbox("Color By", ['None'] + df.columns.tolist(), key="box_color")
        if st.button("Generate Box Plot"):
            color = None if b_color == 'None' else b_color
            fig = create_boxplot(df, b_y, color_by=color)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Bar Chart")
        bar_x = st.selectbox("X Column", df.columns.tolist(), key="bar_x")
        bar_y = st.selectbox("Y Column (optional)", ['None'] + numeric_cols, key="bar_y")
        if st.button("Generate Bar Chart"):
            fig = create_bar_chart(df, bar_x, None if bar_y == 'None' else bar_y)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("### Scatter Plot")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            s_x = st.selectbox("X", numeric_cols, key="scat_x")
        with s_col2:
            s_y = st.selectbox("Y", numeric_cols, key="scat_y")
        with s_col3:
            s_color = st.selectbox("Color", ['None'] + df.columns.tolist(), key="scat_color")
        with s_col4:
            s_size = st.selectbox("Size", ['None'] + numeric_cols, key="scat_size")
        if st.button("Generate Scatter"):
            fig = create_scatter_plot(df, s_x, s_y, None if s_color == 'None' else s_color, None if s_size == 'None' else s_size)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Facet Plot")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            f_x = st.selectbox("X", numeric_cols, key="fac_x")
        with f_col2:
            f_y = st.selectbox("Y", numeric_cols, key="fac_y")
        with f_col3:
            f_col = st.selectbox("Facet", categorical_cols, key="fac_col")
        if st.button("Generate Facet Plot"):
            fig = create_facet_plot(df, f_x, f_y, f_col)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        if datetime_cols:
            st.markdown("### Line Chart")
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                l_x = st.selectbox("Date Column", datetime_cols, key="line_x")
            with l_col2:
                l_y = st.selectbox("Value Column", numeric_cols, key="line_y")
            if st.button("Generate Line Chart"):
                fig = create_line_chart(df, l_x, l_y)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Animated Time Series")
            a_col1, a_col2, a_col3 = st.columns(3)
            with a_col1:
                a_x = st.selectbox("X", numeric_cols, key="anim_x")
            with a_col2:
                a_y = st.selectbox("Y", numeric_cols, key="anim_y")
            with a_col3:
                a_anim = st.selectbox("Animation", datetime_cols + categorical_cols, key="anim_frame")
            if st.button("Generate Animated Plot"):
                fig = create_animated_scatter(df, a_x, a_y, a_anim)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No datetime columns found for time series visualization")

    with tabs[4]:
        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Heatmap")
            selected = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
            if len(selected) >= 2:
                fig = create_correlation_heatmap(df, selected)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Correlation Matrix")
                st.dataframe(df[selected].corr().round(2), use_container_width=True)

            st.markdown("### Pair Plot")
            pair_cols = st.multiselect("Columns for pair plot (max 6)", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], max_selections=6)
            if len(pair_cols) >= 2:
                sample = st.slider("Sample Size", 100, 5000, 1000)
                fig = create_pair_plot(df, pair_cols, sample)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns")

    with tabs[5]:
        st.markdown("### Advanced Visualizations")

        adv_type = st.selectbox("Chart Type", [
            '3D Scatter', 'Treemap', 'Sunburst', 'Waterfall', 'Parallel Categories', 'Density Heatmap'
        ])

        if adv_type == '3D Scatter' and len(numeric_cols) >= 3:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x3d = st.selectbox("X", numeric_cols, key="3d_x")
            with c2:
                y3d = st.selectbox("Y", numeric_cols, key="3d_y")
            with c3:
                z3d = st.selectbox("Z", numeric_cols, key="3d_z")
            with c4:
                c3d = st.selectbox("Color", ['None'] + df.columns.tolist(), key="3d_c")
            if st.button("Generate 3D Scatter"):
                fig = create_3d_scatter(df, x3d, y3d, z3d, None if c3d == 'None' else c3d)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == 'Treemap' and categorical_cols:
            t_path = st.multiselect("Path columns", categorical_cols, default=categorical_cols[:2])
            t_val = st.selectbox("Value", numeric_cols)
            if st.button("Generate Treemap"):
                fig = create_treemap(df, t_path, t_val)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == 'Sunburst' and categorical_cols:
            s_path = st.multiselect("Path columns", categorical_cols, default=categorical_cols[:2])
            s_val = st.selectbox("Value", numeric_cols)
            if st.button("Generate Sunburst"):
                fig = create_sunburst(df, s_path, s_val)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == 'Waterfall':
            w_cat = st.selectbox("Category", categorical_cols if categorical_cols else df.columns.tolist())
            w_val = st.selectbox("Value", numeric_cols)
            if st.button("Generate Waterfall"):
                df_grouped = df.groupby(w_cat)[w_val].sum().reset_index()
                fig = create_waterfall(df_grouped, w_cat, w_val)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == 'Parallel Categories' and len(categorical_cols) >= 2:
            p_cols = st.multiselect("Columns", categorical_cols, default=categorical_cols[:min(5, len(categorical_cols))])
            if len(p_cols) >= 2:
                fig = create_parallel_categories(df, p_cols)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == 'Density Heatmap' and len(numeric_cols) >= 2:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                d_x = st.selectbox("X", numeric_cols, key="dens_x")
            with d_col2:
                d_y = st.selectbox("Y", numeric_cols, key="dens_y")
            if st.button("Generate Density Heatmap"):
                fig = create_density_heatmap(df, d_x, d_y)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.markdown("### Radar Chart")
        r_cols = st.multiselect("Select columns for radar chart", numeric_cols, default=numeric_cols[:5], max_selections=10)
        if len(r_cols) >= 3:
            # Aggregate data for radar chart
            agg_data = df[r_cols].mean().tolist()
            fig = create_radar_chart(df, r_cols, agg_data, "Average Values")
            st.plotly_chart(fig, use_container_width=True)


def render_insights_page():
    """Render smart insights page"""
    df = st.session_state.df
    column_info = st.session_state.column_types

    render_header("💡 Smart Insights", "Automated data analysis and findings")

    if st.button("🔄 Refresh Insights"):
        st.session_state.insights = generate_insights(df, column_info)
        st.rerun()

    render_insights(st.session_state.insights)

    st.markdown("---")
    st.markdown("### 📊 Automated Analysis")

    tabs = st.tabs(["Distribution Analysis", "Normality Tests", "Correlation Analysis"])

    with tabs[0]:
        numeric_cols = get_columns_by_type(column_info, 'numeric')
        for col in numeric_cols[:5]:
            if df[col].nunique() > 5:
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig = create_histogram(df, col, bins=30, kde=True)
                    fig.update_layout(height=300, margin=dict(t=30, b=30, l=30, r=30))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown(f"**{col}**")
                    st.write(f"- Mean: {df[col].mean():.2f}")
                    st.write(f"- Median: {df[col].median():.2f}")
                    st.write(f"- Std: {df[col].std():.2f}")
                    st.write(f"- Skew: {df[col].skew():.2f}")

    with tabs[1]:
        st.markdown("### Normality Test (Shapiro-Wilk)")
        normality_results = test_normality(df)
        st.dataframe(normality_results, use_container_width=True)

        for _, row in normality_results.iterrows():
            if row['is_normal']:
                st.success(f"**{row['column']}**: Appears normally distributed (p={row['p_value']:.4f})")
            else:
                st.warning(f"**{row['column']}**: Not normally distributed (p={row['p_value']:.4f})")

    with tabs[2]:
        st.markdown("### Pairwise Correlation Analysis")
        corr_results = run_all_correlation_tests(df)
        st.dataframe(corr_results, use_container_width=True)

        # Highlight significant correlations
        significant = corr_results[corr_results['significant']]
        if len(significant) > 0:
            st.markdown("### Statistically Significant Correlations")
            for _, row in significant.iterrows():
                strength = 'strong' if abs(row['correlation']) > 0.7 else 'moderate' if abs(row['correlation']) > 0.4 else 'weak'
                direction = 'positive' if row['correlation'] > 0 else 'negative'
                st.write(f"- **{row['col1']}** and **{row['col2']}**: {strength} {direction} (r={row['correlation']:.3f}, p={row['p_value']:.4f})")


def render_target_page():
    """Render target analysis page"""
    df = st.session_state.df
    column_info = st.session_state.column_types

    render_header("🎯 Target Analysis", "Analyze target variable relationships")

    target_col = st.selectbox("Select Target Column", df.columns.tolist(), key="target_select")
    st.session_state.target_column = target_col

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Unique Values", df[target_col].nunique())
        st.metric("Missing Values", df[target_col].isnull().sum())
    with c2:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            st.metric("Mean", f"{df[target_col].mean():.2f}")
            st.metric("Median", f"{df[target_col].median():.2f}")

    # Distribution
    st.markdown("### Target Distribution")
    if pd.api.types.is_numeric_dtype(df[target_col]):
        fig = create_histogram(df, target_col, bins=50, kde=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = create_bar_chart(df, target_col)
        st.plotly_chart(fig, use_container_width=True)

    # Feature vs Target
    st.markdown("### Feature vs Target Relationships")

    numeric_cols = get_columns_by_type(column_info, 'numeric')
    categorical_cols = get_columns_by_type(column_info, 'categorical')

    tabs = st.tabs(["Numeric Features", "Categorical Features"])

    with tabs[0]:
        if numeric_cols:
            feature_col = st.selectbox("Select Feature", numeric_cols, key="target_num")
            c1, c2 = st.columns(2)
            with c1:
                fig = create_scatter_plot(df, feature_col, target_col)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    bins = st.slider("Bins", 3, 20, 5)
                    df_temp = df.copy()
                    df_temp['target_binned'] = pd.cut(df_temp[target_col], bins=bins)
                    grouped = df_temp.groupby('target_binned')[feature_col].agg(['mean', 'std', 'count'])
                else:
                    grouped = df.groupby(target_col)[feature_col].agg(['mean', 'std', 'count'])
                st.dataframe(grouped, use_container_width=True)

    with tabs[1]:
        if categorical_cols:
            feature_col = st.selectbox("Select Feature", categorical_cols, key="target_cat")
            fig = create_boxplot(df, target_col, color_by=feature_col)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Contingency Table")
            crosstab = pd.crosstab(df[feature_col], df[target_col])
            st.dataframe(crosstab, use_container_width=True)

            # Chi-square test
            if st.button("Run Chi-Square Test"):
                chi_result = chi_square_test(df, feature_col, target_col)
                render_statistical_test_results(chi_result)

    # Correlation with target
    if pd.api.types.is_numeric_dtype(df[target_col]) and len(numeric_cols) > 1:
        st.markdown("### Correlation with Target")
        corr_cols = [c for c in numeric_cols if c != target_col]
        corr = df[corr_cols + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)

        def color_bar(val):
            if val > 0.5: return '#EF553B'
            elif val < -0.5: return '#636efa'
            else: return '#999999'

        fig = go.Figure(data=go.Bar(
            x=corr.values,
            y=corr.index,
            orientation='h',
            marker_color=[color_bar(v) for v in corr.values]
        ))
        fig.update_layout(title='Feature Correlations with Target', height=max(300, len(corr) * 30))
        st.plotly_chart(fig, use_container_width=True)


def render_ml_page():
    """Render machine learning page with advanced features"""
    df = st.session_state.df
    column_info = st.session_state.column_types

    render_header("🤖 Machine Learning", "Model training, evaluation, and advanced analytics")

    # Target selection
    st.markdown("### 1. Select Target Variable")
    target_col = st.selectbox("Target Column", df.columns.tolist(), key="ml_target")

    problem_type = determine_problem_type(df, target_col)
    st.info(f"Detected problem type: **{problem_type.upper()}**")

    # Feature selection
    st.markdown("### 2. Select Features")
    available_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select Features", available_cols, default=available_cols[:min(5, len(available_cols))])

    if not selected_features:
        st.warning("Select at least one feature")
        st.stop()

    # Model settings
    st.markdown("### 3. Model Settings")
    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    with c2:
        if problem_type == 'classification':
            model_type = st.selectbox("Model", ['random_forest', 'logistic', 'gradient_boosting', 'svm', 'knn', 'naive_bayes'])
        else:
            model_type = st.selectbox("Model", ['random_forest', 'linear', 'ridge', 'gradient_boosting', 'svm', 'knn'])

    # Advanced options
    with st.expander("Advanced Options"):
        scale_features = st.checkbox("Scale Features", value=False)
        enable_cv = st.checkbox("Enable Cross-Validation", value=True)
        cv_folds = st.slider("CV Folds", 3, 10, 5) if enable_cv else 5
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        enable_ensemble = st.checkbox("Enable Ensemble Comparison", value=False)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                data = prepare_data(df, target_col, selected_features, test_size, scale=scale_features)
                X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

                # Train model
                if problem_type == 'classification':
                    model, metrics, cm = train_classification_model(X_train, X_test, y_train, y_test, model_type)

                    st.markdown("### 4. Results")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with c2:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with c3:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with c4:
                        st.metric("F1 Score", f"{metrics['f1']:.4f}")

                    # Confusion matrix
                    st.markdown("### Confusion Matrix")
                    class_names = data['target_encoder'].classes_.tolist() if data['target_encoder'] else list(np.unique(y_test))
                    render_confusion_matrix(cm, class_names)

                    # Feature importance
                    if 'feature_importance' in metrics:
                        st.markdown("### Feature Importance")
                        fi_df = pd.DataFrame({'Feature': selected_features, 'Importance': metrics['feature_importance']})
                        render_feature_importance(fi_df.sort_values('Importance', ascending=False))

                else:
                    model, metrics = train_regression_model(X_train, X_test, y_train, y_test, model_type)

                    st.markdown("### 4. Results")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("RMSE", f"{metrics['rmse']:.4f}")
                    with c2:
                        st.metric("MSE", f"{metrics['mse']:.4f}")
                    with c3:
                        st.metric("R²", f"{metrics['r2']:.4f}")

                    # Actual vs Predicted
                    st.markdown("### Actual vs Predicted")
                    y_pred = model.predict(X_test)
                    fig = make_subplots(rows=1, cols=2, subplot_titles=('Actual vs Predicted', 'Residuals'))
                    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Perfect', line=dict(dash='dash')), row=1, col=1)
                    residuals = y_test - y_pred
                    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'), row=1, col=2)
                    fig.add_hline(y=0, line_dash='dash', row=1, col=2)
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                    if 'feature_importance' in metrics:
                        st.markdown("### Feature Importance")
                        fi_df = pd.DataFrame({'Feature': selected_features, 'Importance': metrics['feature_importance']})
                        render_feature_importance(fi_df.sort_values('Importance', ascending=False))

                # Cross-validation
                if enable_cv:
                    st.markdown("### Cross-Validation")
                    cv_results = cross_validate(data['X_train'], data['y_train'], model_type, cv_folds, problem_type)
                    render_cross_validation_results(cv_results)

                # Hyperparameter tuning
                if enable_tuning:
                    st.markdown("### Hyperparameter Tuning")
                    with st.spinner("Running grid search..."):
                        best_model, best_params, cv_results = hyperparameter_tuning(
                            data['X_train'], data['y_train'], model_type, problem_type, cv_folds
                        )
                        st.markdown("#### Best Parameters")
                        st.json(best_params)

                # Ensemble comparison
                if enable_ensemble:
                    st.markdown("### Ensemble Model")
                    ensemble_model, ensemble_metrics = train_ensemble_model(
                        X_train, X_test, y_train, y_test, 'voting', problem_type
                    )
                    st.markdown("#### Ensemble Results")
                    for k, v in ensemble_metrics.items():
                        st.metric(k, f"{v:.4f}")

                st.session_state.model_results = {
                    'model': model,
                    'features': selected_features,
                    'target': target_col,
                    'problem_type': problem_type,
                    'metrics': metrics
                }

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Model comparison
    if st.button("📊 Compare All Models"):
        with st.spinner("Comparing models..."):
            data = prepare_data(df, target_col, selected_features, 0.2)
            comparison_df = compare_models(
                data['X_train'], data['X_test'], data['y_train'], data['y_test'], problem_type
            )
            st.markdown("### Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)

            # Highlight best
            if problem_type == 'classification':
                best = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
                st.success(f"Best model: **{best['Model']}** (Accuracy: {best['Accuracy']:.4f})")
            else:
                best = comparison_df.loc[comparison_df['R²'].idxmax()]
                st.success(f"Best model: **{best['Model']}** (R²: {best['R²']:.4f})")

    # Prediction interface
    if st.session_state.model_results:
        st.markdown("---")
        st.markdown("### 🔮 Make Predictions")

        pred_cols = st.columns(len(selected_features))
        input_values = {}

        for i, feature in enumerate(selected_features):
            with pred_cols[i]:
                if df[feature].dtype == 'object' or str(df[feature].dtype) == 'category':
                    input_values[feature] = st.selectbox(feature, df[feature].unique().tolist(), key=f"pred_{feature}")
                else:
                    min_val = float(df[feature].min()) if pd.api.types.is_numeric_dtype(df[feature]) else 0
                    input_values[feature] = st.number_input(feature, value=min_val)

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_values])
                for col in input_df.columns:
                    if df[col].dtype == 'object':
                        le = data['encoders'].get(col)
                        if le:
                            input_df[col] = le.transform(input_df[col].astype(str))

                model = st.session_state.model_results['model']
                prediction = model.predict(input_df)[0]

                if problem_type == 'classification' and data['target_encoder']:
                    prediction = data['target_encoder'].inverse_transform([prediction])[0]

                st.success(f"Predicted: **{prediction}**")
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")


def render_statistics_page():
    """Render statistical analysis page"""
    df = st.session_state.df

    render_header("📐 Statistical Analysis", "Hypothesis testing and advanced statistics")

    tabs = st.tabs(["Descriptive Stats", "Normality", "Group Tests", "Correlation", "PCA"])

    with tabs[0]:
        st.markdown("### Descriptive Statistics")
        stats_df = descriptive_statistics(df)
        st.dataframe(stats_df, use_container_width=True)

    with tabs[1]:
        st.markdown("### Normality Testing (Shapiro-Wilk)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        normality = test_normality(df, numeric_cols)
        st.dataframe(normality, use_container_width=True)

    with tabs[2]:
        st.markdown("### Group Comparison Tests")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            st.warning("No categorical columns found for group comparison tests")
            st.stop()

        num_col = st.selectbox("Numeric Variable", numeric_cols, key="stat_num")
        group_col = st.selectbox("Grouping Variable", categorical_cols, key="stat_group")
        groups = df[group_col].unique().tolist()

        if len(groups) == 2:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("T-Test (Independent)"):
                    result = t_test_independent(df, num_col, group_col, groups[0], groups[1])
                    render_statistical_test_results(result)
            with c2:
                if st.button("Mann-Whitney U"):
                    result = mann_whitney_u(df, num_col, group_col, groups[0], groups[1])
                    render_statistical_test_results(result)
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("ANOVA"):
                    result = anova_test(df, num_col, group_col)
                    render_statistical_test_results(result)
            with c2:
                if st.button("Kruskal-Wallis"):
                    result = kruskal_wallis(df, num_col, group_col)
                    render_statistical_test_results(result)

    with tabs[3]:
        st.markdown("### Correlation Analysis")
        corr_method = st.selectbox("Method", ['pearson', 'spearman', 'kendall'])
        corr_results = run_all_correlation_tests(df, numeric_cols, corr_method)
        st.dataframe(corr_results, use_container_width=True)

        significant = corr_results[corr_results['significant']]
        if len(significant) > 0:
            st.markdown("### Significant Correlations")
            for _, row in significant.iterrows():
                st.write(f"- **{row['col1']}** & **{row['col2']}**: r={row['correlation']:.3f}, p={row['p_value']:.4f}")

    with tabs[4]:
        st.markdown("### Principal Component Analysis")

        pca_cols = st.multiselect("Columns for PCA", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])
        variance_threshold = st.slider("Variance Threshold", 0.5, 0.99, 0.95)

        if st.button("Run PCA"):
            pca_results = pca_analysis(df, pca_cols, variance_threshold=variance_threshold)
            render_pca_results(pca_results)


def render_export_page():
    """Render export page"""
    df = st.session_state.df
    column_info = st.session_state.column_types
    insights = st.session_state.insights
    quality_score = st.session_state.data_quality_score

    render_header("📥 Export", "Download data, reports, and configurations")

    # Data export
    st.markdown("### Export Dataset")
    create_download_buttons(df, "cleaned_data")

    # Summary report
    st.markdown("### Export Summary Report")
    report = generate_summary_report(df, column_info, insights, quality_score)

    st.download_button(
        label="📥 Download Summary (Markdown)",
        data=report.encode('utf-8'),
        file_name=f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True
    )

    # Export package
    st.markdown("### Complete Export Package")
    if st.button("Generate Export Package"):
        package = create_export_package(df, column_info, insights, quality_score, st.session_state.model_results)

        for filename, data in package.items():
            st.download_button(
                label=f"📥 Download {filename}",
                data=data,
                file_name=filename,
                mime="text/plain" if filename.endswith('.md') or filename.endswith('.json') else "text/csv",
                use_container_width=True
            )

    # Session config
    st.markdown("### Export Session Configuration")
    config = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'quality_score': quality_score
        },
        'column_types': column_info,
        'insights_count': len(insights)
    }

    st.download_button(
        label="📥 Download Config (JSON)",
        data=json.dumps(config, indent=2, default=str).encode('utf-8'),
        file_name=f"session_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point"""

    sections = [
        ("📤", "Upload"),
        ("📊", "Overview"),
        ("🧹", "Cleaning"),
        ("🔍", "Filters"),
        ("📈", "Visualizations"),
        ("💡", "Insights"),
        ("🎯", "Target"),
        ("🤖", "ML"),
        ("📐", "Statistics"),
        ("📥", "Export")
    ]

    current_section = render_sidebar([(icon, name) for icon, name in sections])

    page_map = {
        "Upload": render_upload_page,
        "Overview": render_overview_page,
        "Cleaning": render_cleaning_page,
        "Filters": render_filters_page,
        "Visualizations": render_visualizations_page,
        "Insights": render_insights_page,
        "Target": render_target_page,
        "ML": render_ml_page,
        "Statistics": render_statistics_page,
        "Export": render_export_page
    }

    page_map.get(current_section, render_upload_page)()


if __name__ == "__main__":
    main()
