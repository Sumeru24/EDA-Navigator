"""
Visualization Module
Plotly-based visualization functions for EDA.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional, List


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    kde: bool = False
) -> go.Figure:
    """Create histogram with optional KDE"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )

    # Histogram
    hist = go.Histogram(
        x=df[column], nbinsx=bins,
        name='Histogram', marker_color='#636efa'
    )
    fig.add_trace(hist, row=1, col=1)

    # KDE
    if kde and df[column].nunique() > 5:
        x = np.linspace(df[column].min(), df[column].max(), 100)
        kde_data = stats.gaussian_kde(df[column].dropna())
        y = kde_data(x)
        kde_trace = go.Scatter(
            x=x, y=y, name='KDE',
            line=dict(color='#EF553B', width=2)
        )
        fig.add_trace(kde_trace, row=2, col=1)

    fig.update_layout(
        height=500, showlegend=True,
        title_text=f'Distribution of {column}'
    )
    fig.update_xaxes(title_text=column)
    fig.update_yaxes(title_text='Count', row=1, col=1)
    fig.update_yaxes(title_text='Density', row=2, col=1)

    return fig


def create_boxplot(
    df: pd.DataFrame,
    column: str,
    color_by: Optional[str] = None
) -> go.Figure:
    """Create boxplot for outlier detection"""
    if color_by and color_by in df.columns:
        fig = px.box(
            df, x=color_by, y=column, color=color_by,
            title=f'{column} by {color_by}',
            points='all',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        fig = px.box(
            df, y=column, title=f'Boxplot: {column}',
            points='all', color_discrete_sequence=['#636efa']
        )

    fig.update_layout(height=500, showlegend=False)
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None
) -> go.Figure:
    """Create scatter plot with optional coloring and sizing"""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, size=size_col,
        title=f'{x_col} vs {y_col}',
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500)
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create line chart for time series"""
    fig = px.line(
        df, x=x_col, y=y_col, color=color_col,
        title=f'{y_col} over {x_col}',
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(height=500, xaxis_title=x_col, yaxis_title=y_col)
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create bar chart for categorical data"""
    if y_col:
        fig = px.bar(
            df, x=x_col, y=y_col, color=color_col,
            title=f'{y_col} by {x_col}',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
    else:
        df_grouped = df[x_col].value_counts().reset_index()
        df_grouped.columns = [x_col, 'count']
        fig = px.bar(
            df_grouped, x=x_col, y='count',
            title=f'Count by {x_col}',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

    fig.update_layout(height=500, xaxis_title=x_col)
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    numeric_cols: List[str]
) -> go.Figure:
    """Create correlation heatmap"""
    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title='Correlation Heatmap',
        height=600,
        xaxis_title='',
        yaxis_title=''
    )

    return fig


def create_pair_plot(
    df: pd.DataFrame,
    numeric_cols: List[str],
    sample_size: int = 1000
) -> go.Figure:
    """Create pair plot for numeric columns (sampled if large)"""
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    fig = px.scatter_matrix(
        df_sample, dimensions=numeric_cols[:6],  # Limit to 6 columns
        title='Pair Plot (Scatter Matrix)',
        opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_layout(height=600)

    return fig


def create_facet_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    facet_col: str
) -> go.Figure:
    """Create faceted plot"""
    fig = px.scatter(
        df, x=x_col, y=y_col, facet_col=facet_col,
        title=f'{y_col} vs {x_col} (Faceted by {facet_col})',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500)

    return fig


# ============================================================================
# ADVANCED VISUALIZATIONS
# ============================================================================

def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create 3D scatter plot"""
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df[color_col] if color_col else '#636efa',
            colorscale='Viridis' if color_col else None,
            opacity=0.8
        )
    )])

    fig.update_layout(
        title=f'3D Scatter: {x_col} vs {y_col} vs {z_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=700
    )

    return fig


def create_violin_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create violin plot for distribution comparison"""
    fig = px.violin(
        df, x=x_col, y=y_col, color=color_col,
        title=f'{y_col} Distribution by {x_col}',
        box=True,
        points='all',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(height=600)
    return fig


def create_treemap(
    df: pd.DataFrame,
    path_cols: List[str],
    value_col: str
) -> go.Figure:
    """Create treemap for hierarchical data visualization"""
    fig = px.treemap(
        df,
        path=path_cols,
        values=value_col,
        title=f'Hierarchical View: {value_col}',
        color=value_col,
        color_continuous_scale='RdBu'
    )

    fig.update_layout(height=600)
    return fig


def create_sunburst(
    df: pd.DataFrame,
    path_cols: List[str],
    value_col: str
) -> go.Figure:
    """Create sunburst chart for hierarchical proportions"""
    fig = px.sunburst(
        df,
        path=path_cols,
        values=value_col,
        title=f'Sunburst: {value_col} Distribution',
        color=value_col,
        color_continuous_scale='RdBu'
    )

    fig.update_layout(height=600)
    return fig


def create_animated_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    animation_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None
) -> go.Figure:
    """Create animated scatter plot over time/categories"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        animation_frame=animation_col,
        color=color_col,
        size=size_col,
        title=f'{x_col} vs {y_col} (Animated by {animation_col})',
        hover_data=df.columns.tolist(),
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        height=600,
        xaxis=dict(range=[df[x_col].min(), df[x_col].max()]),
        yaxis=dict(range=[df[y_col].min(), df[y_col].max()])
    )

    return fig


def create_waterfall(
    df: pd.DataFrame,
    category_col: str,
    value_col: str
) -> go.Figure:
    """Create waterfall chart for cumulative effects"""
    # Calculate cumulative values
    df_sorted = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    df_sorted['cumulative'] = df_sorted[value_col].cumsum()
    df_sorted['base'] = df_sorted['cumulative'] - df_sorted[value_col]

    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        x=df_sorted[category_col],
        y=df_sorted[value_col],
        base=df_sorted['base'],
        name=value_col,
        marker_color=px.colors.qualitative.Set2
    ))

    # Add cumulative line
    fig.add_trace(go.Scatter(
        x=df_sorted[category_col],
        y=df_sorted['cumulative'],
        name='Cumulative',
        line=dict(color='#EF553B', width=3)
    ))

    fig.update_layout(
        title=f'Waterfall: {value_col} by {category_col}',
        barmode='relative',
        height=500
    )

    return fig


def create_parallel_categories(
    df: pd.DataFrame,
    columns: List[str],
    color_col: Optional[str] = None
) -> go.Figure:
    """Create parallel categories plot for multivariate categorical data"""
    fig = px.parallel_categories(
        df,
        dimensions=columns,
        color=color_col if color_col else df[columns[0]],
        title='Parallel Categories Plot',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=600)
    return fig


def create_density_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: Optional[str] = None
) -> go.Figure:
    """Create 2D density heatmap"""
    if z_col:
        fig = px.density_heatmap(
            df, x=x_col, y=y_col, z=z_col,
            title=f'Density: {x_col} vs {y_col} (weighted by {z_col})',
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.density_heatmap(
            df, x=x_col, y=y_col,
            title=f'Density: {x_col} vs {y_col}',
            color_continuous_scale='Viridis'
        )

    fig.update_layout(height=500)
    return fig


def create_radar_chart(
    df: pd.DataFrame,
    categories: List[str],
    values: List[float],
    title: str = 'Radar Chart'
) -> go.Figure:
    """Create radar/spider chart"""
    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title=title,
        height=500
    )

    return fig
