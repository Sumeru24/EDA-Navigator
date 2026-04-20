"""
Reporting Module
Generate PDF reports, dashboards, and export configurations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import base64
from io import BytesIO


def generate_summary_report(
    df: pd.DataFrame,
    column_info: Dict,
    insights: List[Dict],
    quality_score: Dict
) -> str:
    """Generate a markdown summary report"""

    report = f"""# Data Analysis Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total Rows | {len(df):,} |
| Total Columns | {len(df.columns)} |
| Memory Usage | {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB |
| Duplicate Rows | {df.duplicated().sum():,} |

---

## Data Quality Assessment

**Overall Score:** {quality_score['total']}/100 (Grade: {quality_score['grade']})

### Score Breakdown

| Dimension | Score | Weight |
|-----------|-------|--------|
| Completeness | {quality_score['breakdown'].get('completeness', 0):.1f} | 30% |
| Uniqueness | {quality_score['breakdown'].get('uniqueness', 0):.1f} | 20% |
| Variance | {quality_score['breakdown'].get('variance', 0):.1f} | 25% |
| Consistency | {quality_score['breakdown'].get('consistency', 0):.1f} | 25% |

---

## Column Summary

"""

    for col, info in column_info.items():
        report += f"""### {col}
- **Type:** {info['type']}
- **Data Type:** {info['dtype']}
- **Unique Values:** {info['unique_count']:,}
- **Missing Values:** {info['null_count']:,} ({info['null_ratio']*100:.1f}%)
- **High Cardinality:** {'Yes' if info['is_high_cardinality'] else 'No'}

"""

    report += """---

## Key Insights

"""

    if insights:
        for insight in insights:
            icon = {'warning': '⚠️', 'info': 'ℹ️', 'success': '✅'}.get(insight['type'], '📌')
            report += f"- {icon} **{insight['title']}**: {insight['description']}\n"
    else:
        report += "No specific insights generated.\n"

    report += f"""
---

## Statistical Summary

"""

    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report += "### Numeric Columns\n\n"
        report += df[numeric_cols].describe().to_markdown()
        report += "\n\n"

    # Categorical columns summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        report += "### Categorical Columns\n\n"
        for col in cat_cols[:10]:  # Limit to 10
            top_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A'
            report += f"- **{col}**: {df[col].nunique()} unique values, most common: {top_val}\n"

    return report


def create_dashboard_config(
    charts: List[Dict],
    layout: str = 'grid_2x2',
    title: str = 'Custom Dashboard'
) -> Dict:
    """
    Create a dashboard configuration.

    Args:
        charts: List of chart configurations
        layout: Layout type
        title: Dashboard title

    Returns:
        Dashboard configuration dictionary
    """
    return {
        'title': title,
        'layout': layout,
        'created_at': datetime.now().isoformat(),
        'charts': charts
    }


def save_dashboard_config(config: Dict, filename: str) -> str:
    """Save dashboard configuration to JSON"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    return filename


def load_dashboard_config(filename: str) -> Dict:
    """Load dashboard configuration from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def export_figure_to_image(fig: go.Figure, format: str = 'png', scale: int = 2) -> bytes:
    """Convert Plotly figure to image bytes"""
    img_bytes = BytesIO()
    fig.write_image(img_bytes, format=format, scale=scale)
    img_bytes.seek(0)
    return img_bytes.read()


def create_export_package(
    df: pd.DataFrame,
    column_info: Dict,
    insights: List[Dict],
    quality_score: Dict,
    model_results: Optional[Dict] = None
) -> Dict[str, bytes]:
    """
    Create a complete export package with all analysis results.

    Returns dictionary of filename -> bytes
    """
    package = {}

    # Data export
    package['data.csv'] = df.to_csv(index=False).encode('utf-8')

    # Summary report
    report = generate_summary_report(df, column_info, insights, quality_score)
    package['summary_report.md'] = report.encode('utf-8')

    # Configuration
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
    package['session_config.json'] = json.dumps(config, indent=2, default=str).encode('utf-8')

    # Model results if available
    if model_results:
        model_summary = {
            'model_type': model_results.get('problem_type', 'unknown'),
            'features': model_results.get('features', []),
            'target': model_results.get('target', '')
        }
        package['model_config.json'] = json.dumps(model_summary, indent=2).encode('utf-8')

    return package


def create_email_report_body(
    df: pd.DataFrame,
    quality_score: Dict,
    key_metrics: Dict,
    attachments: List[str] = None
) -> str:
    """Create HTML email body for report distribution"""

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ background-color: #1f2937; color: white; padding: 20px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f3f4f6; border-radius: 8px; }}
            .quality-score {{ font-size: 24px; font-weight: bold; color: {'#10b981' if quality_score['total'] >= 80 else '#f59e0b' if quality_score['total'] >= 60 else '#ef4444'}; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f3f4f6; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Data Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <h2>Key Metrics</h2>
    """

    for label, value in key_metrics.items():
        html += f'<div class="metric"><strong>{label}</strong><br>{value}</div>'

    html += f"""
        <h2>Data Quality</h2>
        <p>Overall Score: <span class="quality-score">{quality_score['total']}/100 (Grade: {quality_score['grade']})</span></p>

        <h2>Dataset Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Rows</td><td>{len(df):,}</td></tr>
            <tr><td>Columns</td><td>{len(df.columns)}</td></tr>
            <tr><td>Duplicate Rows</td><td>{df.duplicated().sum():,}</td></tr>
        </table>
    """

    if attachments:
        html += f"""
        <h2>Attachments</h2>
        <ul>
        """
        for att in attachments:
            html += f"<li>{att}</li>"
        html += "</ul>"

    html += """
        <hr>
        <p><em>This report was automatically generated by DataInsight Pro</em></p>
    </body>
    </html>
    """

    return html


def schedule_report_config(
    report_name: str,
    frequency: str,
    recipients: List[str],
    format: str = 'pdf',
    include_attachments: bool = True
) -> Dict:
    """
    Create a scheduled report configuration.

    Args:
        report_name: Name of the report
        frequency: 'daily', 'weekly', 'monthly'
        recipients: List of email addresses
        format: Export format
        include_attachments: Whether to include data attachments

    Returns:
        Schedule configuration dictionary
    """
    cron_expressions = {
        'daily': '0 9 * * *',  # 9 AM daily
        'weekly': '0 9 * * 1',  # 9 AM every Monday
        'monthly': '0 9 1 * *'  # 9 AM on the 1st of each month
    }

    return {
        'report_name': report_name,
        'cron': cron_expressions.get(frequency, '0 9 * * *'),
        'frequency': frequency,
        'recipients': recipients,
        'format': format,
        'include_attachments': include_attachments,
        'created_at': datetime.now().isoformat()
    }


def create_comparison_report(
    datasets: Dict[str, pd.DataFrame],
    comparison_metrics: List[str] = None
) -> str:
    """
    Create a comparison report for multiple datasets.

    Args:
        datasets: Dictionary of dataset_name -> DataFrame
        comparison_metrics: Metrics to compare (default: basic stats)

    Returns:
        Markdown comparison report
    """
    if comparison_metrics is None:
        comparison_metrics = ['rows', 'columns', 'memory', 'duplicates', 'completeness']

    report = f"""# Dataset Comparison Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Datasets Compared:** {', '.join(datasets.keys())}

---

## Summary Comparison

| Metric | """ + ' | '.join(datasets.keys()) + """ |
|--------|""" + '---|' * len(datasets) + """
"""

    # Rows
    report += f"| Rows | {' | '.join(f'{len(df):,}' for df in datasets.values())} |\n"

    # Columns
    report += f"| Columns | {' | '.join(str(len(df.columns)) for df in datasets.values())} |\n"

    # Memory
    report += f"| Memory (MB) | {' | '.join(f'{df.memory_usage(deep=True).sum() / 1024**2:.2f}' for df in datasets.values())} |\n"

    # Duplicates
    report += f"| Duplicates | {' | '.join(f'{df.duplicated().sum():,}' for df in datasets.values())} |\n"

    # Completeness
    completeness_vals = []
    for df in datasets.values():
        total_cells = len(df) * len(df.columns)
        missing = df.isnull().sum().sum()
        completeness_vals.append(f'{(1 - missing/total_cells)*100:.1f}%' if total_cells > 0 else 'N/A')
    report += f"| Completeness | {' | '.join(completeness_vals)} |\n"

    report += """
---

## Individual Dataset Details

"""

    for name, df in datasets.items():
        report += f"""### {name}

- **Rows:** {len(df):,}
- **Columns:** {len(df.columns)}
- **Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- **Duplicate Rows:** {df.duplicated().sum():,}
- **Missing Values:** {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.1f}%)

"""

    return report
