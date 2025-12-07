import streamlit as st
import pandas as pd
from ml_functions import *

st.set_option("client.showErrorDetails", True)

st.title("📊 ML Healthcare – Data Analysis & Clustering App")
st.write("Upload a CSV dataset to start exploring and analyzing your data.")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    info = load_and_preview_data(uploaded_file)

    if info is None:
        st.error("The uploaded CSV file is empty or invalid!")
        st.stop()

    # Display dataset info
    st.subheader("📌 Dataset Preview")
    st.write(info['head'])
    st.write(f"Shape: {info['shape']}")
    st.write("Column data types:")
    st.write(info['dtypes'])
    st.write("Unique values per column:")
    st.write(info['nunique'])

    # Load dataframe for further analysis
    uploaded_file.seek(0)  # reset pointer
    df = pd.read_csv(uploaded_file)

    # 1️⃣ Missing values
    st.subheader("🚨 Missing Values Summary")
    missing = calculate_missing_values(df)
    st.write(missing)

    # 2️⃣ Descriptive stats
    st.subheader("📈 Descriptive Statistics")
    stats = get_descriptive_stats(df)
    st.write(stats)

    # 3️⃣ Clean data
    st.subheader("🧹 Data Cleaning")
    df_clean, clean_report = clean_data(df)
    st.write(clean_report)

    # 4️⃣ Preprocessing
    st.subheader("⚙️ Preprocessing")
    X_preprocessed, feature_names = preprocess_data(df_clean)
    st.success("Preprocessing completed!")
    st.write(f"Number of features after preprocessing: {X_preprocessed.shape[1]}")

    # 5️⃣ PCA
    st.subheader("📉 PCA Analysis")
    pca_data = perform_pca(X_preprocessed)
    st.write(f"Number of components: {pca_data['n_components']}")
    st.write("Explained variance ratio (first 10):")
    st.write(pca_data['explained_variance'][:10])

    # 6️⃣ K-Means clustering
    st.subheader("🧬 K-Means Clustering")
    kmeans_data = perform_kmeans(pca_data["Z"])
    st.write(f"Best K (highest silhouette): {kmeans_data['best_k']}")
    st.write("Silhouette scores per K:")
    st.write(dict(zip(kmeans_data['k_range'], kmeans_data['silhouette_scores'])))

    # 7️⃣ Cluster profiling
    st.subheader("📂 Cluster Profiles")
    profiles = profile_clusters(df_clean, kmeans_data["labels"])
    st.write(profiles)

else:
    st.info("Please upload a CSV file to begin.")
