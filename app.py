import streamlit as st
import os
from data_cleaner import DataPreprocessor
from generate_dummy_data import create_messy_dataset

# --- Page Config ---
st.set_page_config(page_title="AutoPrep Pipeline", page_icon="⚙️", layout="wide")

# --- Session State Initialization ---
if 'data_path' not in st.session_state:
    st.session_state['data_path'] = None

# --- Title & Intro ---
st.title("⚙️ AutoPrep Pipeline Dashboard")
st.markdown("""
**Automated Data Cleaning & Preprocessing Tool.**
*Matches 'Walkthrough: Data cleaning and preprocessing' specifications.*
""")

st.divider()

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("1. Data Source")

    # Option A: Upload
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

    st.write("--- OR ---")

    # Option B: Generate Dummy Data
    if st.button("🎲 Generate Dummy Data"):
        create_messy_dataset()  # Calls your script to make 'raw_data.csv'
        st.session_state['data_path'] = 'raw_data.csv'  # Store the path in memory
        st.success("Dummy data generated!")
        uploaded_file = None

    st.header("2. Pipeline Settings")
    use_imputation = st.checkbox("Handle Missing (Mean/Mode)", value=True)
    use_outliers = st.checkbox("Remove Outliers (Z-Score < 3)", value=True)

    # Updated: Selection for Scaling Method
    scaler_type = st.radio(
        "Scaling Method",
        ["StandardScaler (Z-Score)", "MinMax Scaler (0-1)"],
        index=0,
        help="StandardScaler is required for the Walkthrough activity."
    )

    use_encoding = st.checkbox("Encode Categories (One-Hot)", value=True)

    # --- VERSION INDICATOR ---
    st.markdown("---")
    st.caption("v1.1.0 | AutoPrep Pipeline")

# --- Logic: Determine which file to use ---
active_path = None

if uploaded_file is not None:
    active_path = "temp_upload.csv"
    with open(active_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['data_path'] = active_path

elif st.session_state['data_path'] is not None:
    active_path = st.session_state['data_path']

# --- Main Dashboard ---
if active_path and os.path.exists(active_path):

    # 1. Initialize Pipeline
    pipeline = DataPreprocessor(active_path)
    pipeline.load_data()

    # 2. Show Raw Data
    st.subheader("📊 Raw Data Preview")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info(f"Rows: {pipeline.df.shape[0]}\nColumns: {pipeline.df.shape[1]}")
    with col2:
        st.dataframe(pipeline.df, height=300)

        # 3. The Action Button
    st.write("")
    if st.button("🚀 Run Preprocessing Pipeline", type="primary"):

        with st.spinner("Processing..."):
            if use_imputation:
                pipeline.handle_missing_values(strategy='fill_mean')

            if use_outliers:
                pipeline.handle_outliers()

            # Updated Scaling Logic
            method = 'standard' if "StandardScaler" in scaler_type else 'minmax'
            pipeline.scale_data(method=method)

            if use_encoding:
                pipeline.encode_categorical()

        # 4. Show Results
        st.divider()
        st.subheader("✅ Processed Data Results")

        col3, col4 = st.columns([1, 3])
        with col3:
            st.success(f"Rows: {pipeline.df.shape[0]}\nColumns: {pipeline.df.shape[1]}")
            st.write("**Verification:**")
            st.text(f"Missing Values: {pipeline.df.isnull().sum().sum()}")
            st.text(f"Columns: {list(pipeline.df.columns)}")
        with col4:
            st.dataframe(pipeline.df, height=300)

        # 5. Download Button
        output_file = "preprocessed_dummy_data.csv"
        pipeline.save_data(output_file)

        with open(output_file, "rb") as file:
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=file,
                file_name="preprocessed_dummy_data.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Waiting for data. Please upload a file or click 'Generate Dummy Data'.")