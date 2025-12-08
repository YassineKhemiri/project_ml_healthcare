import streamlit as st
import pandas as pd
import numpy as np
from ml_functions import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_option("client.showErrorDetails", True)
st.title("📊 ML Healthcare Interactive Dashboard")

# Sidebar
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
variance_threshold = st.sidebar.slider("PCA variance threshold", 0.5, 0.99, 0.8, 0.05)

manual_k = st.sidebar.number_input(
    "Manual K for K-Means (optional, 0 = automatic)",
    min_value=0,
    max_value=10,
    value=0
)

# Load dataset
if uploaded_file is not None:

    info = load_and_preview_data(uploaded_file)
    if info is None:
        st.error("The uploaded CSV file is empty or invalid!")
        st.stop()

    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Data Preview")
    st.write(df.head())

    # Missing values
    st.subheader("🚨 Missing Values")
    st.write(calculate_missing_values(df))

    # Descriptive stats
    st.subheader("📈 Descriptive Statistics")
    st.write(get_descriptive_stats(df))

    # Visualizations
    st.subheader("📊 Histograms & Boxplots")
    if st.button("Generate Histograms"):
        for fig in plot_histograms(df):
            st.pyplot(fig)

    if st.button("Generate Boxplots"):
        for fig in plot_boxplots(df):
            st.pyplot(fig)

    # Cleaning
    st.subheader("🧹 Data Cleaning")
    df_clean, clean_report = clean_data(df)
    st.write(clean_report)

    # Preprocessing
    st.subheader("⚙️ Preprocessing")
    X_preprocessed, feature_names, fitted_preprocessor = preprocess_data(df_clean)
    st.success(f"Preprocessing completed: {X_preprocessed.shape[1]} features")

    # PCA
    st.subheader("📉 PCA Analysis")
    pca_data = perform_pca(X_preprocessed, variance_threshold=variance_threshold)
    st.write(f"Number of components: {pca_data['n_components']}")
    st.pyplot(plot_pca_scatter(np.array(pca_data['Z'])))

    # K-Means
    st.subheader("🧬 K-Means Clustering")

    Z = np.array(pca_data["Z"])
    pca_model = pca_data["pca_model"]

    if manual_k >= 2:
        kmeans = KMeans(n_clusters=manual_k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(Z)
        st.write(f"Manual K selected: {manual_k}")
        st.pyplot(plot_pca_scatter(Z, labels))
    else:
        kmeans_data = perform_kmeans(Z)
        kmeans = kmeans_data["model"]
        labels = kmeans_data["labels"]
        st.write(f"Best K (highest silhouette): {kmeans_data['best_k']}")
        st.pyplot(plot_pca_scatter(Z, labels))

    # Cluster profiling
    st.subheader("📂 Cluster Profiles")
    st.write(profile_clusters(df_clean, labels))

    # New sample input
    st.subheader("✏️ Test New Sample")
    st.write("Enter values for numeric columns:")
    new_data = {}

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        new_data[col] = st.number_input(
            f"{col}", 
            value=float(df_clean[col].median())
        )

    if st.button("Predict Cluster for New Sample"):

        # 1. Preprocess new sample
        X_new_pre = prepare_single_data_point(df_clean, new_data, preprocessor=fitted_preprocessor)

        # 2. Apply PCA
        X_new_pca = pca_model.transform(X_new_pre)

        # 3. Predict with KMeans
        cluster_pred = kmeans.predict(X_new_pca)

        st.success(f"Predicted cluster: {cluster_pred[0]}")

else:
    st.info("Please upload a CSV file to start.")
