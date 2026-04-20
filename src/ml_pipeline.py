"""
Machine Learning Pipeline Module
Model training, evaluation, and prediction with advanced features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve
)
from typing import Tuple, Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def determine_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """Determine if classification or regression"""
    target_data = df[target_col]
    unique_count = target_data.nunique()

    if unique_count <= 10 or target_data.dtype == 'object':
        return 'classification'
    else:
        return 'regression'


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = False
) -> Dict[str, Any]:
    """
    Prepare data for machine learning.

    Args:
        df: Input DataFrame
        target_col: Target column name
        feature_cols: List of feature columns (default: all except target)
        test_size: Test set proportion
        random_state: Random seed
        scale: Whether to scale features

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, encoders, scaler
    """
    df_ml = df.copy()

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X = df_ml[feature_cols].copy()
    y = df_ml[target_col].copy()

    # Encode categorical features
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Encode target if classification
    target_encoder = None
    if y.dtype == 'object' or str(y.dtype) == 'category':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y.astype(str))

    # Drop rows with any NaN
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]
    y = y[mask]

    # Scale features if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # Split data
    stratify = y if determine_problem_type(df_ml, target_col) == 'classification' and len(np.unique(y)) <= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'encoders': encoders,
        'target_encoder': target_encoder,
        'scaler': scaler
    }


def train_classification_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'random_forest'
) -> Tuple[Any, Dict, np.ndarray]:
    """Train classification model"""
    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42, multi_class='auto'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB()
    }

    model = models.get(model_type, models['random_forest'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    if hasattr(model, 'feature_importances_'):
        metrics['feature_importance'] = model.feature_importances_

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return model, metrics, cm


def train_regression_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'random_forest'
) -> Tuple[Any, Dict]:
    """Train regression model"""
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'svm': SVR(kernel='rbf'),
        'knn': KNeighborsRegressor(n_neighbors=5)
    }

    model = models.get(model_type, models['random_forest'])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mae': np.mean(np.abs(y_test - y_pred))
    }

    if hasattr(model, 'feature_importances_'):
        metrics['feature_importance'] = model.feature_importances_

    return model, metrics


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = 'random_forest',
    cv: int = 5,
    problem_type: str = 'classification'
) -> Dict[str, float]:
    """
    Perform cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model to use
        cv: Number of folds
        problem_type: 'classification' or 'regression'

    Returns:
        Dictionary with cross-validation scores
    """
    models_class = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42, multi_class='auto'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB()
    }

    models_reg = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'svm': SVR(kernel='rbf'),
        'knn': KNeighborsRegressor(n_neighbors=5)
    }

    if problem_type == 'classification':
        model = models_class.get(model_type, models_class['random_forest'])
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    else:
        model = models_reg.get(model_type, models_reg['random_forest'])
        scoring = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring[0])

    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'scores': scores.tolist()
    }


def hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = 'random_forest',
    problem_type: str = 'classification',
    cv: int = 5
) -> Tuple[Any, Dict, Dict]:
    """
    Perform hyperparameter tuning with GridSearchCV.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model
        problem_type: 'classification' or 'regression'
        cv: Number of CV folds

    Returns:
        Tuple of (best_model, best_params, cv_results)
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'logistic': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2']
        },
        'svm': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }

    if problem_type == 'classification':
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'svm': SVC(random_state=42),
            'knn': KNeighborsClassifier()
        }
        scoring = 'accuracy'
    else:
        base_models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'ridge': Ridge(),
            'svm': SVR(),
            'knn': KNeighborsRegressor()
        }
        scoring = 'r2'

    base_model = base_models.get(model_type, base_models['random_forest'])
    param_grid = param_grids.get(model_type, param_grids['random_forest'])

    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def train_ensemble_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    ensemble_type: str = 'voting',
    problem_type: str = 'classification'
) -> Tuple[Any, Dict]:
    """
    Train ensemble model (voting or stacking).

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        ensemble_type: 'voting' or 'stacking'
        problem_type: 'classification' or 'regression'

    Returns:
        Tuple of (ensemble_model, metrics)
    """
    if problem_type == 'classification':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]

        if ensemble_type == 'voting':
            model = VotingClassifier(estimators=estimators, voting='soft')
        else:  # stacking
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42),
                cv=5
            )
    else:
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('lr', LinearRegression())
        ]

        if ensemble_type == 'voting':
            model = VotingRegressor(estimators=estimators)
        else:  # stacking
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=LinearRegression(),
                cv=5
            )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    else:
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

    return model, metrics


def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    problem_type: str = 'classification'
) -> pd.DataFrame:
    """
    Compare multiple models and return comparison table.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        problem_type: 'classification' or 'regression'

    Returns:
        DataFrame with model comparison results
    """
    results = []

    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            })
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'KNN': KNeighborsRegressor()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                'Model': name,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MSE': mean_squared_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred),
                'MAE': np.mean(np.abs(y_test - y_pred))
            })

    return pd.DataFrame(results)


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0]) if len(model.coef_.shape) == 1 else np.abs(model.coef_).mean(axis=0)
    else:
        return None

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return fi_df


def calculate_roc_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: Optional[List] = None
) -> Dict:
    """Calculate ROC AUC metrics for classification"""
    if len(np.unique(y_test)) == 2:  # Binary
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'type': 'binary'
        }
    else:  # Multiclass
        fpr, tpr, roc_auc = {}, {}, {}
        n_classes = len(np.unique(y_test))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        return {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'type': 'multiclass',
            'n_classes': n_classes
        }
