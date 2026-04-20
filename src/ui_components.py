"""
UI Components Module
Premium, modern Streamlit UI components with enhanced visual design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================

def inject_premium_css():
    """Inject premium CSS styling for modern UI"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main container */
    .main > div {
        padding-top: 1rem;
    }

    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        color: white;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card.success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    .metric-card.warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
    }
    .metric-card.info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }

    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.75rem;
        opacity: 0.8;
        margin-top: 0.25rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        border-left: 4px solid #667eea;
    }
    .section-header h2 {
        margin: 0;
        color: white;
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f3f4f6;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: white;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Expander styling */
    .stExpander {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
        border: 2px solid transparent;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
    }
    .sidebar-content {
        color: #f9fafb;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Insight cards */
    .insight-card {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .insight-card.warning {
        border-left-color: #f5576c;
        background: linear-gradient(90deg, #fef2f2 0%, white 100%);
    }
    .insight-card.info {
        border-left-color: #4facfe;
        background: linear-gradient(90deg, #f0f9ff 0%, white 100%);
    }
    .insight-card.success {
        border-left-color: #38ef7d;
        background: linear-gradient(90deg, #ecfdf5 0%, white 100%);
    }
    .insight-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .insight-desc {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-grade-a { background: #38ef7d; color: #064e3b; }
    .badge-grade-b { background: #a7f3d0; color: #065f46; }
    .badge-grade-c { background: #fef3c7; color: #92400e; }
    .badge-grade-d { background: #fed7aa; color: #c2410c; }
    .badge-grade-f { background: #fecaca; color: #991b1b; }

    /* Divider */
    .fancy-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# CORE COMPONENTS
# ============================================================================

def render_header(title: str, subtitle: str = "", icon: str = "📊"):
    """Render premium page header"""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown(f"# {icon} {title}")
        if subtitle:
            st.markdown(f"<p style='color: #6b7280; margin-top: -1.5rem; font-size: 1.1rem;'>{subtitle}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown("")
        st.markdown("")


def render_premium_metric(label: str, value: str, delta: str = None, icon: str = "📊", card_type: str = "default"):
    """Render a single premium metric card"""
    card_class = f"metric-card {card_type}"
    delta_html = f"<div class='metric-sub'>{delta}</div>" if delta else ""

    st.markdown(f"""
    <div class="{card_class}">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(metrics: Dict[str, Any], cols: int = 5):
    """Render premium metric cards in columns"""
    columns = st.columns(min(len(metrics), cols))

    metric_icons = {
        'rows': '📄', 'columns': '📊', 'memory': '💾',
        'quality': '⭐', 'duplicates': '🔄', 'score': '🎯',
        'completeness': '✅', 'uniqueness': '🔒', 'variance': '📈', 'consistency': '⚖️'
    }

    for i, (label, value) in enumerate(metrics.items()):
        with columns[i % len(columns)]:
            icon = metric_icons.get(label.lower(), '📊')

            # Determine card type based on label
            card_type = "default"
            if 'quality' in label.lower() or 'score' in label.lower():
                card_type = "success"
            elif 'duplicate' in label.lower() or 'missing' in label.lower():
                card_type = "warning"
            elif 'complete' in label.lower() or 'unique' in label.lower():
                card_type = "info"

            # Extract just the numeric part for display
            display_value = str(value)
            if isinstance(value, float):
                display_value = f"{value:.1f}" if value < 100 else f"{value:.0f}"

            st.markdown(f"""
            <div class="metric-card {card_type}">
                <div class="metric-label">{icon} {label.replace('_', ' ').title()}</div>
                <div class="metric-value">{display_value}</div>
            </div>
            """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = "📌"):
    """Render a styled section header"""
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon} {title}</h2>
    </div>
    """, unsafe_allow_html=True)


def render_data_quality_breakdown(quality_score: Dict):
    """Render data quality score with visual breakdown"""
    # Overall score badge
    grade = quality_score.get('grade', 'C')
    total = quality_score.get('total', 0)

    grade_colors = {'A': '#38ef7d', 'B': '#10b981', 'C': '#f59e0b', 'D': '#f97316', 'F': '#ef4444'}
    grade_color = grade_colors.get(grade, '#6b7280')

    # Display overall score
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, {grade_color}20 0%, {grade_color}40 100%); border-radius: 12px; border: 2px solid {grade_color};">
            <div style="font-size: 3rem; font-weight: 700; color: {grade_color};">{grade}</div>
            <div style="color: #6b7280; font-size: 0.9rem;">Quality Grade</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### Score Breakdown")
        breakdown = quality_score.get('breakdown', {})

        for key, value in breakdown.items():
            max_score = {'completeness': 30, 'uniqueness': 20, 'variance': 25, 'consistency': 25}.get(key, 25)
            percentage = (value / max_score) * 100

            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500;">{key.replace('_', ' ').title()}</span>
                    <span style="color: #6b7280;">{value:.1f}/{max_score}</span>
                </div>
                <div style="background: #e5e7eb; border-radius: 10px; height: 8px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, {grade_color} 0%, {grade_color}aa 100%); width: {percentage}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_insight_card(insight: Dict):
    """Render a single insight card"""
    icons = {'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}
    icon = icons.get(insight['type'], '📌')

    st.markdown(f"""
    <div class="insight-card {insight['type']}">
        <div class="insight-title">{icon} {insight['title']}</div>
        <div class="insight-desc">{insight['description']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_insights(insights: List[Dict]):
    """Render categorized insights with premium cards"""
    warnings_list = [i for i in insights if i['type'] == 'warning']
    info_list = [i for i in insights if i['type'] == 'info']
    success_list = [i for i in insights if i['type'] == 'success']

    if warnings_list:
        render_section_header(f"⚠️ Warnings ({len(warnings_list)})", "🚨")
        for insight in warnings_list:
            render_insight_card(insight)

    if info_list:
        render_section_header(f"ℹ️ Information ({len(info_list)})", "📋")
        for insight in info_list:
            render_insight_card(insight)

    if success_list:
        render_section_header(f"✅ Positive Findings ({len(success_list)})", "🎉")
        for insight in success_list:
            render_insight_card(insight)

    if not insights:
        st.success("✨ No issues found! Your data looks clean and well-structured.")


def render_sidebar(sections: List[tuple]) -> str:
    """Render premium sidebar navigation"""
    with st.sidebar:
        # Logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid #374151; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <h1 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">DataInsight Pro</h1>
            <p style="color: #9ca3af; font-size: 0.85rem; margin: 0.5rem 0 0 0;">Enterprise EDA & ML Platform</p>
        </div>
        """, unsafe_allow_html=True)

        # Theme toggle (simplified for premium look)
        st.markdown("### ⚙️ Settings")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = 'light'
        with col2:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = 'dark'

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")
        for icon_name, section in sections:
            disabled = st.session_state.df is None and section != "Upload"

            # Add emoji spacing
            button_label = f"{icon_name} {section}"

            if st.button(button_label, use_container_width=True, disabled=disabled):
                st.session_state.current_section = section

        st.markdown("---")

        # Data info card
        if st.session_state.df is not None:
            st.markdown("### 📁 Current Dataset")
            st.markdown(f"""
            <div style="background: #374151; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #9ca3af;">Rows</span>
                    <span style="color: white; font-weight: 600;">{len(st.session_state.df):,}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #9ca3af;">Columns</span>
                    <span style="color: white; font-weight: 600;">{len(st.session_state.df.columns)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="position: fixed; bottom: 1rem; left: 1rem; right: 1rem; text-align: center; color: #6b7280; font-size: 0.75rem;">
            Built with ❤️ using Streamlit
        </div>
        """, unsafe_allow_html=True)

    return st.session_state.current_section


# ============================================================================
# CHART COMPONENTS
# ============================================================================

def render_chart_container(fig: go.Figure, title: str = "", description: str = ""):
    """Render a chart in a premium container"""
    if title:
        st.markdown(f"#### {title}")
    if description:
        st.markdown(f"<p style='color: #6b7280; margin-top: -0.5rem; margin-bottom: 1rem;'>{description}</p>", unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    """Render premium confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Count")
    ))

    fig.update_layout(
        title='📊 Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(importance_df: pd.DataFrame, top_n: int = 20):
    """Render premium feature importance chart"""
    if importance_df is None or importance_df.empty:
        st.info("📭 No feature importance available for this model")
        return

    top_features = importance_df.head(top_n)

    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='🎯 Top Feature Importances',
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=max(400, len(top_features) * 30),
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cross_validation_results(cv_results: Dict):
    """Render cross-validation results with visual distribution"""
    st.markdown("### 📊 Cross-Validation Results")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Score", f"{cv_results['mean']:.4f}")
    with col2:
        st.metric("Std Dev", f"{cv_results['std']:.4f}")
    with col3:
        st.metric("Best Fold", f"{cv_results['max']:.4f}")
    with col4:
        st.metric("Worst Fold", f"{cv_results['min']:.4f}")

    # Fold scores visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Fold {i+1}' for i in range(len(cv_results['scores']))],
        y=cv_results['scores'],
        marker_color=['#10b981' if s >= cv_results['mean'] else '#f59e0b' for s in cv_results['scores']],
        text=[f'{s:.4f}' for s in cv_results['scores']],
        textposition='auto'
    ))

    fig.update_layout(
        title='Scores by Fold',
        xaxis_title='Fold',
        yaxis_title='Score',
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Fold scores detail
    with st.expander("📋 View Detailed Fold Scores"):
        for i, score in enumerate(cv_results['scores']):
            st.write(f"**Fold {i+1}**: {score:.4f}")


def render_hyperparameter_results(best_params: Dict, cv_results: Dict):
    """Render hyperparameter tuning results"""
    st.markdown("### 🎯 Best Parameters Found")

    # Display best parameters in a nice format
    param_cols = st.columns(min(len(best_params), 3))
    for i, (param, value) in enumerate(best_params.items()):
        with param_cols[i % len(param_cols)]:
            st.markdown(f"""
            <div style="background: #f3f4f6; padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="color: #6b7280; font-size: 0.85rem;">{param}</div>
                <div style="color: #1f2937; font-weight: 700; font-size: 1.1rem;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # Top parameter combinations
    if 'mean_test_score' in cv_results:
        st.markdown("### 📊 Top 5 Parameter Combinations")
        cv_df = pd.DataFrame(cv_results['params'])
        cv_df['mean_score'] = cv_results['mean_test_score']
        top5 = cv_df.nlargest(5, 'mean_score')
        st.dataframe(top5, use_container_width=True)


def render_statistical_test_results(test_results: Dict):
    """Render statistical test results with premium styling"""
    test_stat = test_results.get('statistic') or test_results.get('f_statistic') or \
                test_results.get('chi_square') or test_results.get('t_statistic') or \
                test_results.get('u_statistic') or test_results.get('h_statistic') or \
                test_results.get('shapiro_stat') or test_results.get('correlation', 0)

    p_value = test_results.get('p_value', 0)
    significant = test_results.get('significant', p_value < 0.05)

    # Result card
    result_type = "success" if significant else "info"
    result_icon = "✅" if significant else "ℹ️"
    result_text = "Statistically Significant" if significant else "Not Statistically Significant"

    st.markdown(f"""
    <div class="insight-card {result_type}">
        <div class="insight-title">{result_icon} {result_text}</div>
        <div class="insight-desc">p-value = {p_value:.4f} (α = 0.05)</div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics table
    stats_df = pd.DataFrame({
        'Metric': ['Test Statistic', 'P-Value', 'Significance Level (α)'],
        'Value': [f"{float(test_stat):.4f}", f"{float(p_value):.4f}", "0.05"]
    })
    st.dataframe(stats_df, use_container_width=True)


def render_pca_results(pca_results: Dict):
    """Render PCA analysis results"""
    st.markdown("### 📊 Variance Explained")
    st.dataframe(pca_results['variance_df'], use_container_width=True)

    # Cumulative variance chart
    fig = px.line(
        pca_results['variance_df'],
        x='Component',
        y='Cumulative Variance',
        markers=True,
        title='📈 Cumulative Variance Explained'
    )
    fig.update_traces(marker={'size': 10, 'color': '#667eea'})

    fig.update_layout(
        xaxis_title='Principal Component',
        yaxis_title='Cumulative Variance',
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

    # Loadings heatmap
    st.markdown("### 🔥 Component Loadings")
    fig = go.Figure(data=go.Heatmap(
        z=pca_results['loadings_df'].values,
        x=pca_results['loadings_df'].columns,
        y=pca_results['loadings_df'].index,
        colorscale='RdBu',
        zmid=0,
        text=pca_results['loadings_df'].values.round(2),
        texttemplate='%{text}'
    ))
    fig.update_layout(title='PCA Feature Loadings', height=400)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# UTILITY COMPONENTS
# ============================================================================

def create_download_buttons(df: pd.DataFrame, prefix: str = "data"):
    """Render premium download buttons"""
    from io import BytesIO

    st.markdown("### 📥 Download Options")

    col1, col2 = st.columns(2)

    # CSV Download
    csv_data = df.to_csv(index=False).encode('utf-8')
    with col1:
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Excel Download
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        with col2:
            st.download_button(
                label="📊 Download as Excel",
                data=output.getvalue(),
                file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except ImportError:
        st.warning("Install openpyxl for Excel export: `pip install openpyxl`")


def render_loading_spinner(message: str = "Processing..."):
    """Render a custom loading message"""
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div style="font-size: 2rem; animation: pulse 1.5s infinite;">⚙️</div>
        <p style="color: #6b7280; margin-top: 1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(message: str, icon: str = "📭", action_label: str = None, action_callback=None):
    """Render an empty state with optional action"""
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem; background: #f9fafb; border-radius: 12px;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <p style="color: #6b7280; font-size: 1.1rem;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_callback:
        if st.button(action_label, use_container_width=True):
            action_callback()


def render_comparison_table(results_df: pd.DataFrame, highlight_best: bool = True):
    """Render a styled comparison table"""
    if results_df.empty:
        st.info("📭 No comparison results available")
        return

    # Style the dataframe
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()

    if highlight_best:
        styled = results_df.style.apply(
            lambda x: ['background-color: #d1fae5; color: #065f46; font-weight: 600' if x == x.max() else '' for x in x],
            subset=numeric_cols
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.dataframe(results_df, use_container_width=True)


def render_model_card(model_name: str, metrics: Dict, icon: str = "🤖"):
    """Render a model summary card"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid #667eea40;">
        <h3 style="margin-top: 0;">{icon} {model_name}</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 1rem;">
    """, unsafe_allow_html=True)

    for metric_name, metric_value in metrics.items():
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="color: #6b7280; font-size: 0.85rem;">{metric_name.replace('_', ' ').title()}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">{metric_value:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_premium_ui():
    """Initialize the premium UI styling"""
    inject_premium_css()
