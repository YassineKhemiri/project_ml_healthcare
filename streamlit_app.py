import streamlit as st
import pandas as pd
from ml_functions import *

st.set_option("client.showErrorDetails", True)

st.title("📊 ML Healthcare – Data Analysis & Clustering App")

st.write("Upload a CSV file to start exploring your data.")

# uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    info = load_and_preview_data(uploaded_file)
    df = pd.read_csv(uploaded_file)
    st.subheader("📌 Data Preview")
    st.write(df.head())

    # 1) Basic info
    st.subheader("🔍 Dataset Info")
    info = load_and_preview_data(uploaded_file)
    st.write(info)

    # 2) Missing values
    st.subheader("🚨 Missing Values")
    missing = calculate_missing_values(df)
    st.write(missing)

    # 3) Descriptive stats
    st.subheader("📈 Descriptive Statistics")
    stats = get_descriptive_stats(df)
    st.write(stats)

    # 4) Clean data
    st.subheader("🧹 Data Cleaning")
    df_clean, clean_report = clean_data(df)
    st.write(clean_report)

    # 5) Preprocessing
    st.subheader("⚙️ Preprocessing")
    X_preprocessed, feature_names = preprocess_data(df_clean)
    st.success("Preprocessing completed.")

    # 6) PCA
    st.subheader("📉 PCA Analysis")
    pca_data = perform_pca(X_preprocessed)
    st.write(pca_data)

    # 7) Clustering
    st.subheader("🧬 K-Means Clustering")
    kmeans_data = perform_kmeans(pca_data["Z"])
    st.write(kmeans_data)

    # 8) Cluster profiling
    st.subheader("📂 Cluster Profiles")
    profiles = profile_clusters(df_clean, kmeans_data["labels"])
    st.write(profiles)

else:
    st.info("Please upload a CSV file to begin.")
