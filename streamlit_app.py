import streamlit as st
import pandas as pd
import numpy as np
from ml_functions import (
    load_and_preview_data, calculate_missing_values, get_descriptive_stats,
    clean_data, preprocess_data, perform_pca, perform_kmeans,
    profile_clusters, prepare_single_data_point,
    plot_histograms, plot_boxplots, plot_pca_scatter
)

st.set_option("client.showErrorDetails", True)
st.title("ML Healthcare Interactive Dashboard")

# ======================
# Sidebar
# ======================
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

variance_threshold = st.sidebar.slider(
    "PCA variance threshold", 0.50, 0.99, 0.80, 0.01
)

manual_k = st.sidebar.number_input(
    "Manual K for K-Means (0 = automatic with silhouette)", 
    min_value=0, max_value=15, value=0
)

# ======================
# Main flow
# ======================
if uploaded_file is not None:
    # 1. Load data
    info = load_and_preview_data(uploaded_file)
    if info is None:
        st.error("The uploaded CSV file is empty or invalid!")
        st.stop()

    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(f"Shape: {info['shape']}")
    st.dataframe(pd.DataFrame(info['head']))

    # 2. Missing values & stats
    st.subheader("Missing Values")
    st.table(calculate_missing_values(df_raw))

    st.subheader("Descriptive Statistics")
    stats = get_descriptive_stats(df_raw)
    st.json(stats)  # or use st.tabs for nicer display

    # 3. Visualizations (optional)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Histograms"):
            for fig in plot_histograms(df_raw):
                st.pyplot(fig)
    with col2:
        if st.button("Generate Boxplots"):
            for fig in plot_boxplots(df_raw):
                st.pyplot(fig)

    # 4. Cleaning
    st.subheader("Data Cleaning")
    df_clean, clean_report = clean_data(df_raw.copy())
    st.success("Cleaning done!")
    st.json(clean_report)

    # 5. Preprocessing (OneHot + RobustScaler)
    st.subheader("Preprocessing")
    X_preprocessed, feature_names, preprocessor = preprocess_data(df_clean)
    st.success(f"Preprocessing → {X_preprocessed.shape[1]} features after encoding")

    # 6. PCA
    st.subheader("PCA (Dimensionality Reduction)")
    pca_result = perform_pca(X_preprocessed, variance_threshold=variance_threshold)
    n_comp = pca_result['n_components']
    Z = pca_result['Z']                 # shape (n_samples, n_comp)
    pca_model = pca_result['pca_model']

    st.write(f"Components needed for {variance_threshold:.0%} variance: **{n_comp}**")
    st.pyplot(plot_pca_scatter(Z))

    # 7. K-Means
    # st.subheader("K-Means Clustering")
    # if manual_k >= 2:
    #     kmeans_model = KMeans(n_clusters=manual_k, n_init=20, random_state=42)
    #     labels = kmeans_model.fit_predict(Z)
    #     st.info(f"Manual mode → K = {manual_k}")
    # else:
    #     km_result = perform_kmeans(Z, max_k=10)
    #     kmeans_model = km_result['model']
    #     labels = km_result['labels']
    #     st.info(f"Automatic mode → Best K = {km_result['best_k']} (silhouette)")

    # st.pyplot(plot_pca_scatter(Z, labels))

    # 7. K-Means Clustering
    st.subheader("K-Means Clustering")
    
    if manual_k >= 2:
        kmeans_model = KMeans(n_clusters=manual_k, n_init=20, random_state=42)
        labels = kmeans_model.fit_predict(Z)
        st.info(f"Manual K = {manual_k}")
    else:
        km_result = perform_kmeans(Z, max_k=10)
        kmeans_model = km_result['model']
        labels = km_result['labels']
        st.info(f"Automatic best K = {km_result['best_k']} (silhouette)")
    
    st.pyplot(plot_pca_scatter(Z, labels))

    # 8. Cluster profiling
    st.subheader("Cluster Profiles")
    profile = profile_clusters(df_clean, labels)
    st.write("Cluster sizes:", profile['cluster_sizes'])
    st.write("Numerical means per cluster")
    st.dataframe(pd.DataFrame(profile['numerical_profiles']))
    st.write("Most frequent category per cluster")
    st.json(profile['categorical_profiles'])

    # # 9. Predict new sample
    # st.subheader("Predict Cluster for a New Person")
    # new_data = {}
    # for col in df_clean.select_dtypes(include=[np.number]).columns:
    #     default = float(df_clean[col].median())
    #     new_data[col] = st.number_input(
    #         col, value=default, step=0.1, format="%.2f"
    #     )

    # for col in df_clean.select_dtypes(exclude=[np.number]).columns:
    #     options = [""] + sorted(df_clean[col].dropna().unique().tolist())
    #     new_data[col] = st.selectbox(col, options, index=0)

    # if st.button("Predict Cluster"):
    #     # Step 1: put the new row in the same column order as training data
    #     new_row_preprocessed = prepare_single_data_point(
    #         df_clean, new_data, preprocessor=preprocessor
    #     )                                          # → (1, 22) or whatever

    #     # Step 2: apply the SAME PCA that was used for training
    #     new_row_pca = pca_model.transform(new_row_preprocessed)   # → (1, n_comp)

    #     # Step 3: predict with the SAME KMeans model
    #     predicted_cluster = kmeans_model.predict(new_row_pca)[0]

    #     st.success(f"Predicted Cluster: **{predicted_cluster}**")
    #     st.balloons()
# ——————————————————————————————————————
# 9. Predict cluster for a new person
# ——————————————————————————————————————
st.subheader("Predict Cluster for a New Person")

st.write("Fill in the information below (numeric + categorical fields):")

new_data = {}

# Numeric inputs
for col in df_clean.select_dtypes(include=[np.number]).columns:
    default_val = float(df_clean[col].median())
    new_data[col] = st.number_input(
        col,
        value=default_val,
        step=0.1,
        format="%.2f",
        key=f"num_{col}"
    )

# Categorical inputs
for col in df_clean.select_dtypes(exclude=[np.number]).columns:
    options = sorted(df_clean[col].dropna().unique().tolist())
    new_data[col] = st.selectbox(
        col,
        options=options,
        index=0,
        key=f"cat_{col}"
    )

if st.button("Predict Cluster", type="primary"):
    try:
        # 1. Put the new row in the exact same format as training data
        X_new_preprocessed = prepare_single_data_point(
            df_clean, new_data, preprocessor=preprocessor  # this is the fitted ColumnTransformer
        )  # shape → (1, 22) or however many features after OneHot

        # 2. Apply the SAME PCA that was used during training
        X_new_pca = pca_model.transform(X_new_preprocessed)   # shape → (1, n_components)

        # 3. Predict with the SAME KMeans model
        predicted_cluster = kmeans_model.predict(X_new_pca)[0]

        st.success(f"Predicted Cluster: **{predicted_cluster}**")
        st.balloons()

    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    st.info("Please upload a CSV file (HealthMind_Mental_Health_Data_75k_MultiAlgo.csv) to start.")
