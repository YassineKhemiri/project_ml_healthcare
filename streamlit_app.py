import streamlit as st
import pandas as pd
import numpy as np
from ml_functions import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_option("client.showErrorDetails", True)
st.title("📊 ML Healthcare Interactive Dashboard")

# Sidebar for user options
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
variance_threshold = st.sidebar.slider("PCA variance threshold", min_value=0.5, max_value=0.99, value=0.8, step=0.05)
# manual_k = st.sidebar.number_input("Manual K for K-Means (optional)", min_value=2, max_value=10, value=0)
manual_k = st.sidebar.number_input(
    "Manual K for K-Means (optional, 0 = automatic)", 
    min_value=0, max_value=10, value=0
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
    missing = calculate_missing_values(df)
    st.write(missing)

    # Descriptive stats
    st.subheader("📈 Descriptive Statistics")
    stats = get_descriptive_stats(df)
    st.write(stats)

    # Visualizations
    st.subheader("📊 Histograms & Boxplots")
    if st.button("Generate Histograms"):
        hist_figs = plot_histograms(df)
        for fig in hist_figs:
            st.pyplot(fig)

    if st.button("Generate Boxplots"):
        box_figs = plot_boxplots(df)
        for fig in box_figs:
            st.pyplot(fig)

    # Clean data
    st.subheader("🧹 Data Cleaning")
    df_clean, clean_report = clean_data(df)
    st.write(clean_report)

    # Preprocessing
    st.subheader("⚙️ Preprocessing")
    X_preprocessed, feature_names = preprocess_data(df_clean)
    st.success(f"Preprocessing completed: {X_preprocessed.shape[1]} features")

    # PCA
    st.subheader("📉 PCA Analysis")
    pca_data = perform_pca(X_preprocessed, variance_threshold=variance_threshold)
    st.write(f"Number of components: {pca_data['n_components']}")
    st.pyplot(plot_pca_scatter(np.array(pca_data['Z'])))

    # K-Means
    st.subheader("🧬 K-Means Clustering")
    # if manual_k >= 2:
    #     k = manual_k
    #     kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    #     labels = kmeans.fit_predict(np.array(pca_data['Z']))
    #     st.write(f"Manual K selected: {k}")
    #     st.pyplot(plot_pca_scatter(np.array(pca_data['Z']), labels))
    # else:
    #     kmeans_data = perform_kmeans(np.array(pca_data['Z']))
    #     st.write(f"Best K (highest silhouette): {kmeans_data['best_k']}")
    #     st.pyplot(plot_pca_scatter(np.array(pca_data['Z']), kmeans_data['labels']))

    if manual_k >= 2:
        k = manual_k
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = kmeans.fit_predict(np.array(pca_data['Z']))
        st.write(f"Manual K selected: {k}")
        st.pyplot(plot_pca_scatter(np.array(pca_data['Z']), labels))
    else:
    # automatic K selection
        kmeans_data = perform_kmeans(np.array(pca_data['Z']))
        st.write(f"Best K (highest silhouette): {kmeans_data['best_k']}")
        st.pyplot(plot_pca_scatter(np.array(pca_data['Z']), kmeans_data['labels']))


    # Cluster profiling
    st.subheader("📂 Cluster Profiles")
    if manual_k >= 2:
        profiles = profile_clusters(df_clean, labels)
    else:
        profiles = profile_clusters(df_clean, kmeans_data['labels'])
    st.write(profiles)

    # Manual input for testing new sample
    st.subheader("✏️ Test New Sample")
    st.write("Enter values for numeric columns to test cluster assignment:")
    new_data = {}
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        new_data[col] = st.number_input(f"{col}", value=float(df_clean[col].median()))
    
    if st.button("Predict Cluster for New Sample"):
        # Use the preprocessor from preprocessing
        X_new = prepare_single_data_point(df_clean, new_data, preprocessor=ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), df_clean.select_dtypes(include=['int64','float64']).columns.tolist())
        ]))
        # Predict with KMeans
        if manual_k >= 2:
            cluster_pred = kmeans.predict(X_new)
        else:
            cluster_pred = kmeans_data['labels'][:1]  # just a placeholder
        st.success(f"Predicted cluster: {cluster_pred[0]}")

else:
    st.info("Please upload a CSV file to start.")
