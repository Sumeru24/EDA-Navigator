"""
Advanced Filtering Module
Custom query builder, saved filter presets, and conditional filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import re


class QueryBuilder:
    """Build complex queries using a fluent interface"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.conditions = []

    def add_condition(self, column: str, operator: str, value: Any) -> 'QueryBuilder':
        """
        Add a filter condition.

        Operators: '==', '!=', '>', '<', '>=', '<=', 'in', 'not in',
                   'contains', 'startswith', 'endswith', 'is null', 'is not null'
        """
        self.conditions.append((column, operator, value))
        return self

    def add_range_condition(
        self,
        column: str,
        min_val: Any = None,
        max_val: Any = None,
        inclusive: bool = True
    ) -> 'QueryBuilder':
        """Add a range filter condition"""
        if min_val is not None and max_val is not None:
            if inclusive:
                self.conditions.append((column, 'between', (min_val, max_val)))
            else:
                self.conditions.append((column, 'between_exclusive', (min_val, max_val)))
        elif min_val is not None:
            self.conditions.append((column, '>', min_val))
        elif max_val is not None:
            self.conditions.append((column, '<', max_val))
        return self

    def add_custom_query(self, query_string: str) -> 'QueryBuilder':
        """Add a custom pandas query string"""
        self.conditions.append(('__custom__', 'query', query_string))
        return self

    def apply_and(self) -> pd.DataFrame:
        """Apply all conditions with AND logic"""
        result = self.df.copy()

        for column, operator, value in self.conditions:
            if column == '__custom__':
                result = result.query(value)
                continue

            if operator == '==':
                result = result[result[column] == value]
            elif operator == '!=':
                result = result[result[column] != value]
            elif operator == '>':
                result = result[result[column] > value]
            elif operator == '<':
                result = result[result[column] < value]
            elif operator == '>=':
                result = result[result[column] >= value]
            elif operator == '<=':
                result = result[result[column] <= value]
            elif operator == 'in':
                result = result[result[column].isin(value)]
            elif operator == 'not in':
                result = result[~result[column].isin(value)]
            elif operator == 'contains':
                result = result[result[column].astype(str).str.contains(value, case=False, na_filter=False)]
            elif operator == 'startswith':
                result = result[result[column].astype(str).str.startswith(value, na_filter=False)]
            elif operator == 'endswith':
                result = result[result[column].astype(str).str.endswith(value, na_filter=False)]
            elif operator == 'is null':
                result = result[result[column].isnull()]
            elif operator == 'is not null':
                result = result[result[column].notnull()]
            elif operator == 'between':
                result = result[(result[column] >= value[0]) & (result[column] <= value[1])]
            elif operator == 'between_exclusive':
                result = result[(result[column] > value[0]) & (result[column] < value[1])]

        return result.reset_index(drop=True)

    def apply_or(self) -> pd.DataFrame:
        """Apply all conditions with OR logic"""
        result = pd.DataFrame(index=self.df.index)
        mask = pd.Series([False] * len(self.df), index=self.df.index)

        for column, operator, value in self.conditions:
            if column == '__custom__':
                temp_mask = self.df.query(value).index
                mask = mask | self.df.index.isin(temp_mask)
                continue

            if operator == '==':
                mask = mask | (self.df[column] == value)
            elif operator == '!=':
                mask = mask | (self.df[column] != value)
            elif operator == '>':
                mask = mask | (self.df[column] > value)
            elif operator == '<':
                mask = mask | (self.df[column] < value)
            elif operator == '>=':
                mask = mask | (self.df[column] >= value)
            elif operator == '<=':
                mask = mask | (self.df[column] <= value)
            elif operator == 'in':
                mask = mask | (self.df[column].isin(value))
            elif operator == 'not in':
                mask = mask | (~self.df[column].isin(value))
            elif operator == 'contains':
                mask = mask | (self.df[column].astype(str).str.contains(value, case=False, na_filter=False))
            elif operator == 'startswith':
                mask = mask | (self.df[column].astype(str).str.startswith(value, na_filter=False))
            elif operator == 'endswith':
                mask = mask | (self.df[column].astype(str).str.endswith(value, na_filter=False))
            elif operator == 'is null':
                mask = mask | (self.df[column].isnull())
            elif operator == 'is not null':
                mask = mask | (self.df[column].notnull())
            elif operator == 'between':
                mask = mask | ((self.df[column] >= value[0]) & (self.df[column] <= value[1]))

        return self.df[mask].reset_index(drop=True)

    def clear(self) -> 'QueryBuilder':
        """Clear all conditions"""
        self.conditions = []
        return self

    def get_condition_count(self) -> int:
        """Get number of active conditions"""
        return len(self.conditions)


class FilterPresets:
    """Manage saved filter presets"""

    def __init__(self):
        self.presets = {}

    def save_preset(self, name: str, conditions: List[Tuple]):
        """Save a filter preset"""
        self.presets[name] = conditions

    def load_preset(self, name: str) -> Optional[List[Tuple]]:
        """Load a filter preset"""
        return self.presets.get(name)

    def delete_preset(self, name: str) -> bool:
        """Delete a filter preset"""
        if name in self.presets:
            del self.presets[name]
            return True
        return False

    def list_presets(self) -> List[str]:
        """List all preset names"""
        return list(self.presets.keys())

    def apply_preset(self, name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply a saved preset to a dataframe"""
        conditions = self.presets.get(name)
        if conditions is None:
            return None

        builder = QueryBuilder(df)
        builder.conditions = conditions
        return builder.apply_and()


class ConditionalFormatting:
    """Apply conditional formatting rules to data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.rules = []

    def add_rule(
        self,
        column: str,
        condition: str,
        value: Any,
        format_style: Dict[str, str]
    ) -> 'ConditionalFormatting':
        """
        Add a conditional formatting rule.

        condition: '>', '<', '==', '>=', '<=', '!=', 'between', 'contains', 'top', 'bottom'
        format_style: dict with keys like 'background-color', 'color', 'font-weight'
        """
        self.rules.append((column, condition, value, format_style))
        return self

    def add_highlight_max(self, column: str, color: str = '#d4edda') -> 'ConditionalFormatting':
        """Highlight maximum values in a column"""
        max_val = self.df[column].max()
        self.add_rule(column, '==', max_val, {'background-color': color})
        return self

    def add_highlight_min(self, column: str, color: str = '#f8d7da') -> 'ConditionalFormatting':
        """Highlight minimum values in a column"""
        min_val = self.df[column].min()
        self.add_rule(column, '==', min_val, {'background-color': color})
        return self

    def add_highlight_outliers(
        self,
        column: str,
        method: str = 'iqr',
        color: str = '#fff3cd'
    ) -> 'ConditionalFormatting':
        """Highlight outlier values"""
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.add_rule(column, 'outlier', (lower, upper), {'background-color': color})
        return self

    def add_gradient(self, column: str, cmap: str = 'RdYlGn') -> 'ConditionalFormatting':
        """Add gradient background to numeric column"""
        self.rules.append((column, 'gradient', cmap, {}))
        return self

    def apply(self):
        """Apply all formatting rules and return styled dataframe"""
        styled = self.df.style

        for column, condition, value, format_style in self.rules:
            if condition == 'gradient':
                styled = styled.background_gradient(cmap=value, subset=[column])
            elif condition == 'outlier':
                lower, upper = value
                def highlight_outlier(x):
                    return [f'background-color: {format_style.get("background-color", "#fff3cd")}'
                            if (x < lower or x > upper) else '' for _ in x]
                styled = styled.apply(highlight_outlier, subset=[column])
            else:
                # Create mask for condition
                def apply_format(x, col=column, cond=condition, val=value, fmt=format_style):
                    styles = [''] * len(x)
                    for i, cell_val in enumerate(x):
                        if self._check_condition(cell_val, cond, val):
                            styles[i] = '; '.join(f'{k}: {v}' for k, v in fmt.items())
                    return styles

                styled = styled.apply(apply_format, subset=[column], axis=0)

        return styled

    def _check_condition(self, cell_value: Any, condition: str, compare_value: Any) -> bool:
        """Check if a cell value meets a condition"""
        if pd.isna(cell_value):
            return False

        if condition == '>':
            return cell_value > compare_value
        elif condition == '<':
            return cell_value < compare_value
        elif condition == '>=':
            return cell_value >= compare_value
        elif condition == '<=':
            return cell_value <= compare_value
        elif condition == '==':
            return cell_value == compare_value
        elif condition == '!=':
            return cell_value != compare_value
        elif condition == 'between':
            return compare_value[0] <= cell_value <= compare_value[1]
        elif condition == 'contains':
            return compare_value in str(cell_value)
        elif condition == 'top':
            return cell_value >= self.df[condition].quantile(1 - compare_value)
        elif condition == 'bottom':
            return cell_value <= self.df[condition].quantile(compare_value)

        return False


def create_smart_filter(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Create smart filter suggestions based on column characteristics.

    Returns dict with suggested filters and their parameters.
    """
    suggestions = {}

    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric filters
        col_min, col_max = df[column].min(), df[column].max()
        col_mean = df[column].mean()
        col_std = df[column].std()

        suggestions['range'] = {
            'type': 'range',
            'min': col_min,
            'max': col_max,
            'default': (col_min, col_max)
        }

        suggestions['above_average'] = {
            'type': 'condition',
            'operator': '>',
            'value': col_mean,
            'description': f'Above average (>{col_mean:.2f})'
        }

        suggestions['below_average'] = {
            'type': 'condition',
            'operator': '<',
            'value': col_mean,
            'description': f'Below average (<{col_mean:.2f})'
        }

        suggestions['within_1std'] = {
            'type': 'range',
            'min': col_mean - col_std,
            'max': col_mean + col_std,
            'description': 'Within 1 standard deviation'
        }

        suggestions['outliers_high'] = {
            'type': 'condition',
            'operator': '>',
            'value': col_mean + 3 * col_std,
            'description': 'High outliers (>3σ)'
        }

    elif df[column].dtype == 'object' or str(df[column].dtype) == 'category':
        # Categorical filters
        value_counts = df[column].value_counts()
        top_values = value_counts.head(10).index.tolist()

        suggestions['top_values'] = {
            'type': 'multiselect',
            'options': top_values,
            'description': 'Top 10 values'
        }

        suggestions['non_null'] = {
            'type': 'condition',
            'operator': 'is not null',
            'value': None,
            'description': 'Non-null values only'
        }

    return suggestions


def apply_filters_from_config(df: pd.DataFrame, filter_config: Dict) -> pd.DataFrame:
    """
    Apply filters from a configuration dictionary.

    Config format:
    {
        'logic': 'and' or 'or',
        'filters': [
            {'column': 'col1', 'operator': '>', 'value': 10},
            {'column': 'col2', 'operator': 'in', 'value': ['A', 'B']}
        ]
    }
    """
    builder = QueryBuilder(df)

    for f in filter_config.get('filters', []):
        builder.add_condition(f['column'], f['operator'], f['value'])

    if filter_config.get('logic', 'and') == 'or':
        return builder.apply_or()
    else:
        return builder.apply_and()
