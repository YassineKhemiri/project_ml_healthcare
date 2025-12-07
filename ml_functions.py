import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io
import base64
import re


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# def load_and_preview_data(file_path):
#     df = pd.read_csv(file_path)
#     if 'User_ID' in df.columns:
#         df = df.drop('User_ID', axis=1)
#     shape = df.shape
#     head = df.head(5).to_dict(orient='records')
#     dtypes = df.dtypes.to_dict()
#     nunique = df.nunique().sort_values(ascending=False).to_dict()
#     return {
#         'shape': shape,
#         'head': head,
#         'dtypes': dtypes,
#         'nunique': nunique
#     }

# def load_and_preview_data(file):
#     # If the file is a BytesIO (Streamlit upload), read it directly
#     df = pd.read_csv(file)

#     if 'User_ID' in df.columns:
#         df = df.drop('User_ID', axis=1)

#     shape = df.shape
#     head = df.head(5).to_dict(orient='records')
#     dtypes = df.dtypes.to_dict()
#     nunique = df.nunique().sort_values(ascending=False).to_dict()

#     return {
#         'shape': shape,
#         'head': head,
#         'dtypes': dtypes,
#         'nunique': nunique
#     }

def load_and_preview_data(uploaded_file):
    """
    Load CSV uploaded via Streamlit and return summary information.
    """
    uploaded_file.seek(0)  # reset pointer
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        return None  # Streamlit will handle empty file

    if df.empty:
        return None

    if 'User_ID' in df.columns:
        df = df.drop('User_ID', axis=1)

    shape = df.shape
    head = df.head(5).to_dict(orient='records')
    dtypes = df.dtypes.to_dict()
    nunique = df.nunique().sort_values(ascending=False).to_dict()

    return {
        'shape': shape,
        'head': head,
        'dtypes': dtypes,
        'nunique': nunique
    }


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

def get_distributions(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    dist_data = {}
    for col in num_cols:
        hist, bins = np.histogram(df[col].dropna(), bins=30)
        dist_data[col] = {'bins': bins.tolist(), 'counts': hist.tolist()}
    return dist_data

def get_boxplot_data(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    box_data = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        box_data[col] = {'min': df[col].min(), 'q1': q1, 'median': df[col].median(), 'q3': q3, 'max': df[col].max(), 'iqr': iqr}
    return box_data

def get_risk_distribution(df):
    if 'Risk_Level' in df.columns:
        counts = df['Risk_Level'].value_counts().to_dict()
        percentages = (df['Risk_Level'].value_counts(normalize=True) * 100).round(2).to_dict()
        return {'counts': counts, 'percentages': percentages}
    return None

def calculate_quartiles_iqr(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    iqr_report = []
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        Q1 = s.quantile(0.25)
        Q2 = s.quantile(0.50)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((s < lower_bound) | (s > upper_bound))
        outlier_count = int(outliers.sum())
        outlier_pct = (outlier_count / len(s) * 100)
        iqr_report.append({
            'Variable': col,
            'Q1': Q1,
            'Median': Q2,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outliers_Count': outlier_count,
            'Outliers_Pct': outlier_pct
        })
    return sorted(iqr_report, key=lambda x: x['Outliers_Count'], reverse=True)

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

    if 'Sleep_Hours_Night' in df.columns:
        n_invalid_sleep = (df['Sleep_Hours_Night'] < 0).sum()
        if n_invalid_sleep > 0:
            df.loc[df['Sleep_Hours_Night'] < 0, 'Sleep_Hours_Night'] = np.nan

    if 'Exercise_Freq_Week' in df.columns:
        n_invalid_ex = (df['Exercise_Freq_Week'] > 7).sum()
        if n_invalid_ex > 0:
            df.loc[df['Exercise_Freq_Week'] > 7, 'Exercise_Freq_Week'] = np.nan

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

def calculate_correlations(df):
    drop_cols = []
    if 'User_ID' in df.columns:
        drop_cols.append('User_ID')
    if 'Risk_Level' in df.columns:
        drop_cols.append('Risk_Level')
    X = df.drop(columns=drop_cols, errors='ignore')
    num_only = X.select_dtypes(include=['int64', 'float64'])
    corr_pearson = num_only.corr(method='pearson').to_dict()
    corr_spearman = num_only.corr(method='spearman').to_dict()

    high_corr_pairs = []
    for i in range(len(num_only.columns)):
        for j in range(i+1, len(num_only.columns)):
            if abs(corr_pearson[num_only.columns[i]][num_only.columns[j]]) > 0.8:
                high_corr_pairs.append({
                    'Var1': num_only.columns[i],
                    'Var2': num_only.columns[j],
                    'Correlation': corr_pearson[num_only.columns[i]][num_only.columns[j]]
                })

    return {'pearson': corr_pearson, 'spearman': corr_spearman, 'high_corr_pairs': high_corr_pairs}

# def preprocess_data(df):
#     drop_cols = []
#     if 'User_ID' in df.columns:
#         drop_cols.append('User_ID')
#     if 'Risk_Level' in df.columns:
#         drop_cols.append('Risk_Level')
#     X = df.drop(columns=drop_cols, errors='ignore')

#     num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     cat_features = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

#     num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', RobustScaler())
#     ])

#     cat_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
#     ])

#     preprocessor = ColumnTransformer([
#         ('num', num_pipeline, num_features),
#         ('cat', cat_pipeline, cat_features)
#     ])

#     X_preprocessed = preprocessor.fit_transform(X)
#     feature_names = preprocessor.get_feature_names_out().tolist()

#     return X_preprocessed, feature_names

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

    X_preprocessed = preprocessor.fit_transform(X)  # fit AND transform

    feature_names = preprocessor.get_feature_names_out().tolist()

    # Return the fitted preprocessor so you can use it later on new data
    return X_preprocessed, feature_names, preprocessor


def perform_pca(X_preprocessed, variance_threshold=0.8):
    n_components_max = min(20, X_preprocessed.shape[1], X_preprocessed.shape[0])
    pca_init = PCA(n_components=n_components_max, random_state=RANDOM_STATE)
    Z_init = pca_init.fit_transform(X_preprocessed)

    explained_variance = pca_init.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    n_components_80 = np.argmax(cumulative_variance >= variance_threshold) + 1
    if cumulative_variance[-1] < variance_threshold:
        n_components_80 = len(cumulative_variance)

    pca_final = PCA(n_components=n_components_80, random_state=RANDOM_STATE)
    Z = pca_final.fit_transform(X_preprocessed)

    loadings = pca_final.components_.T * np.sqrt(pca_final.explained_variance_)

    return {
        'n_components': n_components_80,
        'explained_variance': explained_variance.tolist(),
        'cumulative_variance': cumulative_variance.tolist(),
        'Z': Z.tolist(),
        'loadings': loadings.tolist()
    }

def perform_kmeans(Z, max_k=10):
    silhouette_scores = []
    inertias = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(Z)
        score = silhouette_score(Z, labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)

    best_k = np.argmax(silhouette_scores) + 2

    kmeans_final = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans_final.fit_predict(Z)

    return {
        'k_range': list(range(2, max_k + 1)),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias,
        'best_k': best_k,
        'labels': labels,
        'centroids': kmeans_final.cluster_centers_.tolist()
    }

def profile_clusters(df, labels):
    df_copy = df.copy()
    df_copy['cluster'] = labels
    num_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_copy.select_dtypes(exclude=[np.number]).columns.tolist()

    profiles_num = df_copy.groupby('cluster')[num_cols].mean().to_dict(orient='records')
    profiles_cat = {}
    for col in cat_cols:
        profiles_cat[col] = df_copy.groupby('cluster')[col].agg(lambda x: x.mode()[0] if not x.empty else None).to_dict()

    if 'Risk_Level' in df_copy.columns:
        risk_dist = df_copy.groupby('cluster')['Risk_Level'].value_counts(normalize=True).unstack().fillna(0).to_dict(orient='records')
    else:
        risk_dist = None

    cluster_sizes = df_copy['cluster'].value_counts().sort_index().to_dict()

    return {
        'numerical_profiles': profiles_num,
        'categorical_profiles': profiles_cat,
        'risk_distribution': risk_dist,
        'cluster_sizes': cluster_sizes
    }

def generate_plot_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_histograms(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    figs = []
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        figs.append(fig)
    return figs

def plot_boxplots(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    figs = []
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        figs.append(fig)
    return figs

def plot_pca_scatter(Z, labels=None):
    import matplotlib.pyplot as plt
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

# def prepare_single_data_point(df, new_data_dict, preprocessor):
#     """
#     Transform a single new data point using the existing preprocessor pipeline
#     """
#     import pandas as pd
#     new_df = pd.DataFrame([new_data_dict])
#     X_preprocessed = preprocessor.transform(new_df)
#     return X_preprocessed

def prepare_single_data_point(df, new_data_dict, preprocessor):
    """
    Transform a single new data point using the already fitted preprocessor.
    Fills missing columns with np.nan instead of pd.NA.
    """
    import pandas as pd
    import numpy as np

    # Create DataFrame from new data
    new_df = pd.DataFrame([new_data_dict])

    # Ensure all columns from training df exist
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan  # <-- use np.nan, NOT pd.NA

    # Reorder columns to match training data
    new_df = new_df[df.columns]

    # Transform using the fitted preprocessor
    X_preprocessed = preprocessor.transform(new_df)
    return X_preprocessed


