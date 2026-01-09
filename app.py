import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaner import DataPreprocessor
from generate_dummy_data import create_messy_dataset

# --- Page Config ---
st.set_page_config(page_title="Data Quality Control Center", page_icon="🛡️", layout="wide")

if 'data_path' not in st.session_state:
    st.session_state['data_path'] = None

# --- Header ---
st.title("🛡️ Data Quality Control Center")
st.markdown("**Detect, Validate, and Clean Data Collection Errors.** [v1.2.0]")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.write("--- OR ---")
    if st.button("🎲 Generate Messy Data"):
        create_messy_dataset()
        st.session_state['data_path'] = 'raw_data.csv'
        uploaded_file = None
        st.success("Messy data generated!")

    st.header("2. Validation Rules")
    use_duplicates = st.checkbox("Step 5: Remove Duplicates", value=True)
    use_integrity = st.checkbox("Step 4: Fix Logic (Negative Ages)", value=True)
    use_text = st.checkbox("Step 4: Standardize Text", value=True)
    use_math = st.checkbox("Step 5: Fix Calculation Errors", value=True)

    st.header("3. Cleaning Rules")
    use_imputation = st.checkbox("Step 2: Handle Missing Values", value=True)
    use_outliers = st.checkbox("Step 3: Remove Outliers", value=True)
    use_scaling = st.checkbox("Scale Features", value=False)

# --- Load Data ---
active_path = None
if uploaded_file:
    active_path = "temp.csv"
    with open(active_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['data_path'] = active_path
elif st.session_state['data_path']:
    active_path = st.session_state['data_path']

# --- Main Dashboard ---
if active_path and os.path.exists(active_path):
    pipeline = DataPreprocessor(active_path)
    pipeline.load_data()

    # TABS Interface
    tab1, tab2, tab3 = st.tabs(["📊 Data Inspector", "📉 Outlier Diagnostics", "🚀 Cleaning Pipeline"])

    # TAB 1: INSPECTOR
    with tab1:
        st.subheader("Raw Data Inspection")
        st.dataframe(pipeline.df, height=300)

        col1, col2 = st.columns(2)
        with col1:
            st.warning(f"Duplicates Detected: {pipeline.df.duplicated().sum()}")
        with col2:
            neg_ages = (pipeline.df['Age'] < 0).sum() if 'Age' in pipeline.df.columns else 0
            st.error(f"Logic Errors (Negative Ages): {neg_ages}")

    # TAB 2: DIAGNOSTICS (Box Plots)
    with tab2:
        st.subheader("Step 3: Outlier Detection (Box Plots)")
        numeric_cols = pipeline.df.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select Column to Visualize", numeric_cols,
                                        index=len(numeric_cols) - 1)  # Default to last (Salary)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(x=pipeline.df[selected_col], ax=ax, color='orange')
            st.pyplot(fig)
        else:
            st.info("No numeric columns found.")

    # TAB 3: PIPELINE
    with tab3:
        if st.button("🚀 Run Quality Control Pipeline", type="primary"):
            with st.spinner("Validating and Cleaning..."):
                # 1. Validation Steps (New)
                if use_duplicates: pipeline.remove_duplicates()
                if use_integrity: pipeline.validate_integrity()
                if use_text: pipeline.standardize_text()
                if use_math: pipeline.fix_consistency()

                # 2. Cleaning Steps (Old)
                if use_imputation: pipeline.handle_missing_values()
                if use_outliers: pipeline.handle_outliers()
                if use_scaling: pipeline.scale_data()

            st.success("Pipeline Complete!")
            st.dataframe(pipeline.df, height=300)

            # Download
            pipeline.save_data("clean_data.csv")
            with open("clean_data.csv", "rb") as f:
                st.download_button("📥 Download Clean Data", f, "clean_data.csv", "text/csv")

else:
    st.info("Upload data or click Generate to begin.")