# DataInsight Pro - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Module Reference](#module-reference)
5. [Feature Guide](#feature-guide)
6. [Usage Examples](#usage-examples)

---

## Overview

**DataInsight Pro** is a production-grade Exploratory Data Analysis (EDA) and Machine Learning platform built with Streamlit. It provides a comprehensive web interface for data analysis, visualization, statistical testing, and ML model training.

### Key Capabilities
- 📊 **Data Upload & Profiling** - CSV, Excel, JSON support with automatic type detection
- 🧹 **Data Cleaning** - Handle missing values, outliers, duplicates, and transformations
- 🔍 **Advanced Filtering** - Query builder with AND/OR logic and saved presets
- 📈 **Visualizations** - 15+ chart types including 3D scatter, treemap, sunburst, and animated plots
- 💡 **Smart Insights** - Automated data quality scoring and anomaly detection
- 🤖 **Machine Learning** - Classification, regression, cross-validation, hyperparameter tuning
- 📐 **Statistical Tests** - Normality tests, t-tests, ANOVA, chi-square, correlation analysis
- 📥 **Export** - Data, reports, charts, and session configurations
z
---

## Architecture

### Modular Structure
```
all EDA/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── src/                   # Modular packages
│   ├── __init__.py
│   ├── data_loader.py     # File loading and type detection
│   ├── data_quality.py    # Quality scoring and insights
│   ├── visualizations.py  # Plotly chart functions
│   ├── cleaning.py        # Data transformation functions
│   ├── ml_pipeline.py     # ML training and evaluation
│   ├── statistics.py      # Statistical tests and PCA
│   ├── advanced_filters.py # Query builder and presets
│   ├── reporting.py       # Report generation
│   └── ui_components.py   # Reusable UI components
└── DOCUMENTATION.md       # This file
```

### Design Principles
1. **Separation of Concerns** - Each module handles a specific domain
2. **Reusability** - Functions are modular and can be imported independently
3. **Caching** - Data loading functions use `@st.cache_data` for performance
4. **Session State** - All user data stored in Streamlit session state
5. **Error Handling** - Graceful degradation when dependencies missing

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
cd "C:\Users\sumer\OneDrive\Desktop\100dayML\EDA\all EDA"
pip install -r requirements.txt
```

### Dependencies Explained
| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.28.0 | Web application framework |
| pandas | >=2.0.0 | Data manipulation and analysis |
| numpy | >=1.24.0 | Numerical computing |
| plotly | >=5.17.0 | Interactive visualizations |
| scikit-learn | >=1.3.0 | Machine learning algorithms |
| openpyxl | >=3.1.0 | Excel file support |
| kaleido | >=0.2.0 | Plotly image export |
| statsmodels | >=0.14.0 | Statistical models |
| scipy | >=1.11.0 | Scientific computing |
| tabulate | >=0.9.0 | Table formatting |

### Step 2: Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Module Reference

### 1. data_loader.py

**Purpose:** Handle file uploads and data type detection.

#### Functions

##### `load_csv(file) -> DataFrame`
Load CSV file with optimized settings.
```python
df = load_csv(uploaded_file)
```

##### `load_excel(file) -> DataFrame`
Load Excel file (.xlsx).
```python
df = load_excel(uploaded_file)
```

##### `load_json(file) -> DataFrame`
Load JSON file.
```python
df = load_json(uploaded_file)
```

##### `detect_column_types(df, cardinality_threshold=0.5) -> dict`
Automatically classify columns as numeric, categorical, or datetime.

**Returns:**
```python
{
    'column_name': {
        'type': 'numeric',  # or 'categorical', 'datetime'
        'unique_count': 150,
        'unique_ratio': 0.15,
        'null_count': 5,
        'null_ratio': 0.005,
        'dtype': 'int64',
        'is_high_cardinality': True
    }
}
```

##### `get_columns_by_type(column_info, col_type) -> list`
Get list of columns matching a type.
```python
numeric_cols = get_columns_by_type(column_info, 'numeric')
```

---

### 2. data_quality.py

**Purpose:** Calculate data quality scores and generate automated insights.

#### Functions

##### `calculate_data_quality_score(df, column_info) -> dict`
Calculate overall data quality score (0-100) based on:
- **Completeness (30%)** - Based on missing values
- **Uniqueness (20%)** - Based on duplicate rows
- **Variance (25%)** - Based on constant columns
- **Consistency (25%)** - Based on data type appropriateness

**Returns:**
```python
{
    'total': 85.5,
    'breakdown': {
        'completeness': 28.5,
        'uniqueness': 18.0,
        'variance': 22.0,
        'consistency': 17.0
    },
    'grade': 'B'  # A, B, C, D, or F
}
```

##### `generate_insights(df, column_info) -> list`
Generate automated insights about the dataset.

**Insight Types:**
- **Warning** - High missing values, outliers, duplicates, low variance
- **Info** - Skewed distributions, high cardinality
- **Success** - Strong positive correlations

**Returns:**
```python
[
    {
        'type': 'warning',
        'title': 'High Missing Values: Age',
        'description': '35.0% of values are missing in this column'
    }
]
```

##### `detect_outliers_iqr(df, columns) -> DataFrame`
Detect outliers using IQR method (values beyond Q1-1.5*IQR or Q3+1.5*IQR).

##### `calculate_skewness_kurtosis(df) -> DataFrame`
Calculate skewness and kurtosis for all numeric columns.

---

### 3. visualizations.py

**Purpose:** Create Plotly-based visualizations.

#### Standard Charts

| Function | Description |
|----------|-------------|
| `create_histogram(df, column, bins, kde)` | Histogram with optional KDE curve |
| `create_boxplot(df, column, color_by)` | Box plot for outlier detection |
| `create_scatter_plot(df, x, y, color, size)` | Scatter plot with styling |
| `create_line_chart(df, x, y, color)` | Line chart for time series |
| `create_bar_chart(df, x, y, color)` | Bar chart for categorical data |
| `create_correlation_heatmap(df, cols)` | Correlation matrix heatmap |
| `create_pair_plot(df, cols, sample)` | Scatter matrix (max 6 columns) |
| `create_facet_plot(df, x, y, facet)` | Faceted scatter plots |

#### Advanced Charts

| Function | Description |
|----------|-------------|
| `create_3d_scatter(df, x, y, z, color)` | 3D scatter plot |
| `create_violin_plot(df, x, y, color)` | Violin plot with box overlay |
| `create_treemap(df, path, value)` | Hierarchical treemap |
| `create_sunburst(df, path, value)` | Sunburst chart |
| `create_animated_scatter(df, x, y, animation)` | Animated scatter over time |
| `create_waterfall(df, category, value)` | Waterfall chart |
| `create_parallel_categories(df, cols)` | Parallel categories plot |
| `create_density_heatmap(df, x, y)` | 2D density heatmap |
| `create_radar_chart(df, categories, values)` | Radar/spider chart |

---

### 4. cleaning.py

**Purpose:** Data transformation and feature engineering.

#### Functions

##### Basic Operations
| Function | Description |
|----------|-------------|
| `drop_columns(df, columns)` | Remove specified columns |
| `drop_duplicates(df)` | Remove duplicate rows |
| `rename_column(df, old, new)` | Rename a column |

##### Missing Value Handling
```python
handle_missing_values(df, strategy='mean', columns=None, fill_value=None)
```
**Strategies:** `mean`, `median`, `mode`, `constant`, `drop`

##### Type Conversion
```python
convert_column_type(df, column, new_type)
```
**Types:** `numeric`, `categorical`, `datetime`, `string`

##### Feature Engineering
```python
create_feature(df, name, operation, col1, col2, value)
```
**Operations:** `add`, `subtract`, `multiply`, `divide`, `ratio`, `log`, `square`, `sqrt`, `bin`, `qcut`

##### Outlier Treatment
```python
treat_outliers_iqr(df, columns, method='cap')
```
**Methods:** 
- `cap` - Winsorize (clip to bounds)
- `remove` - Drop outlier rows
- `flag` - Add outlier indicator column

##### Normalization
```python
normalize_column(df, column, method='zscore')
```
**Methods:** `zscore`, `minmax`, `log`, `boxcox`

##### Encoding
```python
encode_categorical(df, columns, method='onehot')
```
**Methods:** `onehot`, `label`, `frequency`

##### Binning
```python
bin_column(df, column, bins=5, method='equal_width')
```
**Methods:** `equal_width`, `equal_freq`

---

### 5. ml_pipeline.py

**Purpose:** Machine learning model training and evaluation.

#### Functions

##### `determine_problem_type(df, target_col) -> str`
Automatically detect if problem is classification or regression.

##### `prepare_data(df, target_col, features, test_size, scale) -> dict`
Prepare data for ML: encode categoricals, split train/test, optionally scale.

**Returns:**
```python
{
    'X_train': DataFrame,
    'X_test': DataFrame,
    'y_train': array,
    'y_test': array,
    'encoders': dict,
    'target_encoder': LabelEncoder,
    'scaler': StandardScaler
}
```

##### Model Training
```python
train_classification_model(X_train, X_test, y_train, y_test, model_type)
train_regression_model(X_train, X_test, y_train, y_test, model_type)
```

**Model Types:**
- **Classification:** `random_forest`, `logistic`, `gradient_boosting`, `svm`, `knn`, `naive_bayes`
- **Regression:** `random_forest`, `linear`, `ridge`, `gradient_boosting`, `svm`, `knn`

**Returns:** `(model, metrics, confusion_matrix)`

##### Cross-Validation
```python
cross_validate(X, y, model_type, cv=5, problem_type)
```
**Returns:** Mean, std, min, max scores across folds.

##### Hyperparameter Tuning
```python
hyperparameter_tuning(X_train, y_train, model_type, problem_type, cv=5)
```
Uses GridSearchCV with predefined parameter grids.

**Returns:** `(best_model, best_params, cv_results)`

##### Ensemble Models
```python
train_ensemble_model(X_train, X_test, y_train, y_test, ensemble_type, problem_type)
```
**Types:** `voting` (soft voting), `stacking` (meta-learner)

##### Model Comparison
```python
compare_models(X_train, X_test, y_train, y_test, problem_type) -> DataFrame
```
Compares all available models and returns comparison table.

---

### 6. statistics.py

**Purpose:** Statistical hypothesis testing and analysis.

#### Normality Tests
```python
test_normality(df, columns) -> DataFrame
```
Shapiro-Wilk test for each numeric column.

#### Correlation Tests
```python
test_correlation_significance(df, col1, col2, method) -> dict
run_all_correlation_tests(df, columns, method) -> DataFrame
```
**Methods:** `pearson`, `spearman`, `kendall`

#### Group Comparison Tests

| Test | Function | Use Case |
|------|----------|----------|
| Independent T-Test | `t_test_independent(df, num, group, g1, g2)` | Compare 2 groups (parametric) |
| Paired T-Test | `t_test_paired(df, col1, col2)` | Paired measurements |
| Mann-Whitney U | `mann_whitney_u(df, num, group, g1, g2)` | Compare 2 groups (non-parametric) |
| ANOVA | `anova_test(df, num, group)` | Compare 3+ groups (parametric) |
| Kruskal-Wallis | `kruskal_wallis(df, num, group)` | Compare 3+ groups (non-parametric) |
| Chi-Square | `chi_square_test(df, col1, col2)` | Test independence of categoricals |

#### Effect Size
```python
calculate_effect_size(df, num, group, g1, g2, effect_type='cohens_d')
```
**Interpretation:** <0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large

#### PCA
```python
pca_analysis(df, columns, n_components, variance_threshold) -> dict
```
Performs PCA and returns components, variance explained, and loadings.

#### Descriptive Statistics
```python
descriptive_statistics(df) -> DataFrame
```
Comprehensive stats: count, mean, std, min, quartiles, max, skewness, kurtosis.

---

### 7. advanced_filters.py

**Purpose:** Advanced filtering and conditional formatting.

#### QueryBuilder Class
Build complex filters with fluent interface.

```python
from src.advanced_filters import QueryBuilder

builder = QueryBuilder(df)
builder.add_condition('Age', '>', 30)
builder.add_condition('City', 'in', ['NYC', 'LA'])
builder.add_range_condition('Salary', min_val=50000, max_val=100000)

# Apply with AND logic
filtered = builder.apply_and()

# Apply with OR logic
filtered = builder.apply_or()
```

**Operators:** `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `not in`, `contains`, `startswith`, `endswith`, `is null`, `is not null`

#### FilterPresets Class
Save and load filter configurations.

```python
presets = FilterPresets()
presets.save_preset('high_value', [('Income', '>', 100000)])
presets.load_preset('high_value')
presets.apply_preset('high_value', df)
```

#### ConditionalFormatting Class
Apply visual formatting to DataFrames.

```python
formatter = ConditionalFormatting(df)
formatter.add_highlight_max('Sales', color='#d4edda')
formatter.add_highlight_outliers('Price', method='iqr')
formatter.add_gradient('Rating', cmap='RdYlGn')

styled = formatter.apply()
```

---

### 8. reporting.py

**Purpose:** Generate reports and export configurations.

#### Functions

##### `generate_summary_report(df, column_info, insights, quality_score) -> str`
Generate markdown summary report with dataset overview, quality scores, and insights.

##### `create_export_package(df, column_info, insights, quality_score, model_results) -> dict`
Create complete export package with data, report, and config files.

##### `create_dashboard_config(charts, layout, title) -> dict`
Create dashboard configuration for saving layouts.

##### `create_email_report_body(df, quality_score, metrics, attachments) -> str`
Generate HTML email body for report distribution.

---

### 9. ui_components.py

**Purpose:** Reusable Streamlit UI components.

#### Components
| Component | Function |
|-----------|----------|
| Header | `render_header(title, subtitle)` |
| Metrics | `render_metric_cards(metrics_dict, cols)` |
| Quality Breakdown | `render_data_quality_breakdown(score)` |
| Insights | `render_insights(insights_list)` |
| Sidebar | `render_sidebar(sections)` |
| Download Buttons | `create_download_buttons(df)` |
| CV Results | `render_cross_validation_results(cv_dict)` |
| Hyperparameters | `render_hyperparameter_results(params, results)` |
| Feature Importance | `render_feature_importance(fi_df)` |
| Confusion Matrix | `render_confusion_matrix(cm, classes)` |
| Statistical Results | `render_statistical_test_results(test_dict)` |
| PCA Results | `render_pca_results(pca_dict)` |

---

## Feature Guide

### 1. Upload Page
- Upload CSV, Excel, or JSON files
- Automatic data type detection
- File metadata display
- Column type preview

### 2. Overview Page
**Tabs:**
- **Summary** - Key metrics, quality breakdown, statistical summary
- **Head** - First 10 rows
- **Tail** - Last 10 rows
- **Sample** - Random sample with adjustable size
- **Types** - Column type information with override capability
- **Statistics** - Comprehensive descriptive statistics

### 3. Cleaning Page
**Tabs:**
- **Drop** - Remove columns and duplicates
- **Missing** - Handle missing values with multiple strategies
- **Transform** - Rename columns, convert types, normalize
- **Features** - Create new features from existing columns
- **Outliers** - Detect and treat outliers (cap/remove/flag)

### 4. Filters Page
**Tabs:**
- **Query Builder** - Add conditions with various operators
- **Presets** - Save and load filter configurations
- **Conditional Formatting** - Highlight cells based on rules

### 5. Visualizations Page
**Tabs:**
- **Distribution** - Histograms, violin plots
- **Comparison** - Box plots, bar charts
- **Relationship** - Scatter plots, facet plots
- **Time Series** - Line charts, animated plots
- **Correlation** - Heatmaps, pair plots
- **Advanced** - 3D scatter, treemap, sunburst, waterfall, parallel categories, density heatmap
- **Custom** - Radar charts

### 6. Insights Page
- Categorized insights (warnings, info, positive findings)
- Distribution analysis with histograms
- Normality test results
- Pairwise correlation analysis

### 7. Target Analysis Page
- Target variable selection and distribution
- Feature vs target relationships
- Contingency tables
- Chi-square tests
- Correlation with target

### 8. Machine Learning Page
- Automatic problem type detection
- Feature selection
- Model selection (6+ algorithms)
- Train/test split configuration
- Cross-validation
- Hyperparameter tuning
- Ensemble models (voting, stacking)
- Model comparison table
- Prediction interface

### 9. Statistics Page
**Tabs:**
- **Descriptive Stats** - Comprehensive statistics
- **Normality** - Shapiro-Wilk tests
- **Group Tests** - T-tests, ANOVA, non-parametric alternatives
- **Correlation** - Pairwise correlation analysis
- **PCA** - Dimensionality reduction

### 10. Export Page
- Download as CSV/Excel
- Summary report (Markdown)
- Complete export package
- Session configuration (JSON)

---

## Usage Examples

### Example 1: Basic Data Analysis Workflow

```python
# 1. Upload data via UI
# 2. Review data quality in Overview tab
# 3. Clean data: remove duplicates, handle missing values
# 4. Apply filters for specific segment
# 5. Generate visualizations
# 6. Review automated insights
# 7. Export cleaned data
```

### Example 2: Machine Learning Pipeline

```python
# 1. Navigate to ML tab
# 2. Select target variable (e.g., 'Churn')
# 3. Select features (predictors)
# 4. Choose model type (Random Forest)
# 5. Enable cross-validation and hyperparameter tuning
# 6. Click "Train Model"
# 7. Review metrics and feature importance
# 8. Use prediction interface for new data
```

### Example 3: Statistical Analysis

```python
# 1. Go to Statistics tab
# 2. Run normality tests to check distributions
# 3. For 2 groups: Run t-test or Mann-Whitney U
# 4. For 3+ groups: Run ANOVA or Kruskal-Wallis
# 5. Check correlation analysis for relationships
# 6. Run PCA for dimensionality reduction
```

### Example 4: Programmatic Usage

```python
from src.data_loader import load_data, detect_column_types
from src.data_quality import calculate_data_quality_score, generate_insights
from src.ml_pipeline import prepare_data, train_classification_model, cross_validate

# Load and analyze data
df = load_csv('data.csv')
column_info = detect_column_types(df)
quality = calculate_data_quality_score(df, column_info)
insights = generate_insights(df, column_info)

# Train model
data = prepare_data(df, 'target', feature_cols=['f1', 'f2', 'f3'])
model, metrics, cm = train_classification_model(
    data['X_train'], data['X_test'], 
    data['y_train'], data['y_test'], 
    'random_forest'
)

# Cross-validate
cv = cross_validate(data['X_train'], data['y_train'], 'random_forest', cv=5)
```

### Example 5: Custom Query Builder

```python
from src.advanced_filters import QueryBuilder

df = pd.read_csv('customers.csv')
builder = QueryBuilder(df)

# Build complex query
builder.add_condition('Age', '>', 25) \
       .add_condition('City', 'in', ['NYC', 'LA', 'Chicago']) \
       .add_condition('Income', '>=', 50000) \
       .add_range_condition('Score', min_val=70, max_val=100)

# Apply filters
filtered = builder.apply_and()
print(f"Found {len(filtered)} matching records")
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'openpyxl'**
```bash
pip install openpyxl
```

**2. Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**3. File not loading**
- Check file encoding (UTF-8 recommended)
- Ensure file is not corrupted
- Try different file format

**4. Charts not rendering**
- Check for empty data
- Verify column types are correct
- Try reducing data size

---

## Contributing

To add new features:
1. Create new function in appropriate module
2. Add imports to `app.py`
3. Update this documentation

---

## License

This project is for educational and commercial use.

---

*Generated: 2026-04-01*
*Version: 2.0 (Refactored)*
