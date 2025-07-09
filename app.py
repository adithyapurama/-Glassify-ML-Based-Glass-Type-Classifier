import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import base64

# === Page Configuration ===
st.set_page_config(
    page_title="Glassify - ML Glass Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Load Model & Scaler ===
@st.cache_resource
def load_model():
    model = joblib.load("gradient_boosting_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# === Glass Type Mapping ===
GLASS_TYPES = {
    1: "Building Windows (Float Processed)",
    2: "Building Windows (Non-Float Processed)",
    3: "Vehicle Windows (Float Processed)",
    5: "Containers",
    6: "Tableware",
    7: "Headlamps"
}

# === Feature Info ===
FEATURE_INFO = {
    'RI': {'name': 'Refractive Index', 'tooltip': 'Light bending ability', 'min': 1.51, 'max': 1.54, 'step': 0.0001, 'format': "%.4f", 'default': 1.52},
    'Na': {'name': 'Sodium', 'tooltip': 'Amount of Na₂O', 'min': 10.0, 'max': 16.0, 'step': 0.01, 'format': "%.2f", 'default': 13.5},
    'Mg': {'name': 'Magnesium', 'tooltip': 'Stabilizer component', 'min': 0.0, 'max': 4.5, 'step': 0.01, 'format': "%.2f", 'default': 2.5},
    'Al': {'name': 'Aluminum', 'tooltip': 'Glass durability', 'min': 0.0, 'max': 3.5, 'step': 0.01, 'format': "%.2f", 'default': 1.5},
    'Si': {'name': 'Silicon', 'tooltip': 'Primary component (SiO₂)', 'min': 69.0, 'max': 75.0, 'step': 0.01, 'format': "%.2f", 'default': 72.0},
    'K': {'name': 'Potassium', 'tooltip': 'Modifier', 'min': 0.0, 'max': 1.5, 'step': 0.001, 'format': "%.3f", 'default': 0.5},
    'Ca': {'name': 'Calcium', 'tooltip': 'Hardness agent', 'min': 7.0, 'max': 14.0, 'step': 0.01, 'format': "%.2f", 'default': 9.0},
    'Ba': {'name': 'Barium', 'tooltip': 'Glare reducer', 'min': 0.0, 'max': 2.5, 'step': 0.01, 'format': "%.2f", 'default': 0.0},
    'Fe': {'name': 'Iron', 'tooltip': 'Coloring agent', 'min': 0.0, 'max': 0.5, 'step': 0.01, 'format': "%.2f", 'default': 0.1},
}

# === Background Setup ===
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image_path = "images.jpg"
bg_base64 = get_base64_image(bg_image_path)

theme = st.sidebar.radio("🎨 Theme", ["🌞 Light", "🌚 Dark"])
if theme == "🌞 Light":
    base_bg = "rgba(255,255,255,0.5)"
    font_color = "#000"
else:
    base_bg = "rgba(0,0,0,0.5)"
    font_color = "#fff"

st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-attachment: fixed;
        color: {font_color};
    }}
    .glass-box {{
        backdrop-filter: blur(10px);
        background-color: {base_bg};
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# === Navigation ===
st.sidebar.title("🔍 Navigation")
mode = st.sidebar.radio("Choose Mode", ["Single Prediction", "Batch Prediction", "Model Info"])

st.markdown("<h1 class='glass-box'>🔬 Glassify: ML-Based Glass Classifier</h1>", unsafe_allow_html=True)

# === Single Prediction ===
if mode == "Single Prediction":
    st.subheader("📥 Enter Chemical Composition")

    if st.button("🔁 Reset Inputs"):
        st.experimental_rerun()

    user_inputs = {}
    for feature, info in FEATURE_INFO.items():
        col1, col2 = st.columns([2, 3])
        with col1:
            user_inputs[feature] = st.number_input(
                f"{info['name']} ({feature})",
                min_value=info['min'], max_value=info['max'],
                value=info['default'], step=info['step'],
                format=info['format'], help=info['tooltip'], key=f"num_{feature}"
            )
        with col2:
            st.slider(
                f"{info['name']} ({feature})",
                min_value=info['min'], max_value=info['max'],
                value=user_inputs[feature], step=info['step'],
                format=info['format'], help=info['tooltip'],
                key=f"slider_{feature}", label_visibility="collapsed"
            )

    if st.button("🔮 Predict Glass Type", type="primary"):
        input_data = np.array([list(user_inputs.values())])
        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0]

        glass_type = GLASS_TYPES.get(prediction, "Unknown")
        st.markdown(f"""
            <div class='glass-box'>
                <h4>🧪 Glass Type: {glass_type} (Type {prediction})</h4>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Confidence Scores")
        confidence_df = pd.DataFrame({
            "Glass Type": [f"{GLASS_TYPES.get(i, f'Type {i}')} (Type {i})" for i in range(1, 8) if i in GLASS_TYPES],
            "Confidence": proba
        }).sort_values("Confidence", ascending=False)

        fig = px.bar(confidence_df, x="Confidence", y="Glass Type", orientation="h", height=400)
        st.plotly_chart(fig, use_container_width=True)

# === Batch Prediction ===
elif mode == "Batch Prediction":
    st.subheader("📁 Upload CSV for Batch Prediction")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(), use_container_width=True)

        missing_cols = [col for col in FEATURE_INFO if col not in df.columns]
        if missing_cols:
            st.warning(f"Missing columns in CSV: {missing_cols}")
        else:
            if st.button("⚡ Predict All"):
                X = scaler.transform(df[list(FEATURE_INFO)])
                preds = model.predict(X)
                df["Predicted Type"] = [f"{GLASS_TYPES.get(i, 'Unknown')} (Type {i})" for i in preds]
                df["Confidence"] = [max(p) for p in model.predict_proba(X)]
                st.success("✅ Predictions Complete")
                st.dataframe(df, use_container_width=True)
                st.download_button("📥 Download Predictions", df.to_csv(index=False), "glass_predictions.csv", "text/csv")

# === Model Info ===
else:
    st.subheader("📄 Model Info")
    st.markdown("A Gradient Boosting Classifier trained on the UCI Glass Dataset using SMOTE for class balancing.")

    st.markdown("### 🧪 Features")
    st.dataframe(pd.DataFrame([
        {'Feature': f, 'Name': v['name'], 'Tooltip': v['tooltip']} for f, v in FEATURE_INFO.items()
    ]), use_container_width=True)

    st.markdown("### 🔖 Glass Type Mapping")
    st.dataframe(pd.DataFrame([
        {'Class ID': k, 'Glass Type': v} for k, v in GLASS_TYPES.items()
    ]), use_container_width=True)

    st.markdown("### 📈 Performance Metrics (Training Results)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy", "89.47%")
    col2.metric("CV Accuracy", "88.32% ± 2.87%")
    col3.metric("Train Accuracy", "98.54%")

# === Footer ===
st.markdown("---")
st.caption("© 2025 Glassify – Streamlit + Gradient Boosting | Developed by Adithya Purama")
