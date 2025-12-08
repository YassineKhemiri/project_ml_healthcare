import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import base64
import joblib
import os

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------------
# Data Loading & Preview
# -------------------------------
def load_and_preview_data(uploaded_file):
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return None

    if df.empty:
        return None

    if 'User_ID' in df.columns:
        df = df.drop('User_ID', axis=1)

    return {
        'shape': df.shape,
        'head': df.head(5).to_dict(orient='records'),
        'dtypes': df.dtypes.to_dict(),
        'nunique': df.nunique().sort_values(ascending=False).to_dict()
    }

# -------------------------------
# Missing Values & Stats
# -------------------------------
def calculate_missing_values(df):
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing_Count': missing_count, 'Missing_Pct': missing_pct})
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    return missing_summary.to_dict(orient='records')

def get_descriptive_stats(df):
    num_stats = df.describe().T.to_dict()
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_stats = {}
    for col in cat_cols:
        cat_stats[col] = {
            'value_counts': df[col].value_counts().to_dict(),
            'unique': df[col].nunique()
        }
    return {'numerical': num_stats, 'categorical': cat_stats}

# -------------------------------
# Data Cleaning
# -------------------------------
def clean_data(df):
    n_before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    n_after = len(df)
    duplicates_removed = n_before - n_after

    rules_clip = {
        "Work_Hours_Week": (0, 100),
        "Sleep_Hours_Night": (0, 12),
        "Exercise_Freq_Week": (0, 7),
        "Stress_Level_Scale": (1, 8),
        "Age": (16, 80),
    }

    clipped = {}
    for col, (lo, hi) in rules_clip.items():
        if col in df.columns:
            n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_clipped > 0:
                df[col] = df[col].clip(lower=lo, upper=hi)
            clipped[col] = n_clipped

    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    if num_cols:
        imputer_num = SimpleImputer(strategy='median')
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if cat_cols:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    return df, {
        'duplicates_removed': duplicates_removed,
        'clipped': clipped,
        'remaining_missing': df.isna().sum().sum()
    }

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_data(df):
    drop_cols = []
    if 'User_ID' in df.columns:
        drop_cols.append('User_ID')
    if 'Risk_Level' in df.columns:
        drop_cols.append('Risk_Level')
    X = df.drop(columns=drop_cols, errors='ignore')

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    X_preprocessed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    return X_preprocessed, feature_names, preprocessor

# -------------------------------
# PCA
# -------------------------------
def perform_pca(X_preprocessed, variance_threshold=0.8):
    n_components_max = min(20, X_preprocessed.shape[1], X_preprocessed.shape[0])
    pca_init = PCA(n_components=n_components_max, random_state=RANDOM_STATE)
    Z_init = pca_init.fit_transform(X_preprocessed)

    cumulative_variance = np.cumsum(pca_init.explained_variance_ratio_)
    n_components_final = np.argmax(cumulative_variance >= variance_threshold) + 1
    pca_final = PCA(n_components=n_components_final, random_state=RANDOM_STATE)
    Z = pca_final.fit_transform(X_preprocessed)

    loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)

    return {
        'n_components': n_components_final,
        'Z': Z,
        'pca_model': pca_final,
        'loadings': loadings.tolist()
    }

# -------------------------------
# KMeans
# -------------------------------
def perform_kmeans(Z, max_k=10):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(Z)
        silhouette_scores.append(silhouette_score(Z, labels))

    best_k = np.argmax(silhouette_scores) + 2
    kmeans_final = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans_final.fit_predict(Z)

    return {
        'best_k': best_k,
        'labels': labels,
        'model': kmeans_final
    }

# -------------------------------
# Cluster Profiling
# -------------------------------
def profile_clusters(df, labels):
    df_copy = df.copy()
    df_copy['cluster'] = labels
    num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()

    profiles_num = df_copy.groupby('cluster')[num_cols].mean().to_dict(orient='records')
    profiles_cat = {col: df_copy.groupby('cluster')[col].agg(lambda x: x.mode()[0] if not x.empty else None).to_dict()
                    for col in cat_cols}

    cluster_sizes = df_copy['cluster'].value_counts().sort_index().to_dict()

    return {
        'numerical_profiles': profiles_num,
        'categorical_profiles': profiles_cat,
        'cluster_sizes': cluster_sizes
    }

# -------------------------------
# New Sample
# -------------------------------
def prepare_single_data_point(df_clean, new_data, preprocessor):
    full_columns = df_clean.columns.tolist()
    row = {col: new_data.get(col, np.nan) for col in full_columns}
    new_df = pd.DataFrame([row])
    X_preprocessed = preprocessor.transform(new_df)
    return X_preprocessed

# -------------------------------
# Plotting
# -------------------------------
def plot_histograms(df):
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        figs.append(fig)
    return figs

def plot_boxplots(df):
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        figs.append(fig)
    return figs

def plot_pca_scatter(Z, labels=None):
    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], c='blue', alpha=0.6)
    else:
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='Set1', alpha=0.6)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Scatter Plot")
    return fig
