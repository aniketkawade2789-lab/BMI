"""
Fitness Classification System — Streamlit App
Uses Random Forest ML to predict fitness category and provide recommendations.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

LABELS = {
    0: {
        "category": "Underfit",
        "emoji": "💪",
        "color": "#2196F3",
        "recommendation": (
            "Increase daily calorie intake with protein-rich foods "
            "(eggs, chicken, legumes). Focus on strength training 3–4x/week "
            "and ensure 7–9 hours of sleep for muscle recovery."
        ),
    },
    1: {
        "category": "Fit",
        "emoji": "✅",
        "color": "#4CAF50",
        "recommendation": (
            "Maintain your balanced diet and consistent exercise routine. "
            "Mix cardio and strength training. Stay hydrated and monitor "
            "progress monthly to keep on track."
        ),
    },
    2: {
        "category": "Overfit",
        "emoji": "🔥",
        "color": "#FF5722",
        "recommendation": (
            "Create a moderate calorie deficit (~300–500 kcal/day). "
            "Prioritize cardio (30 min/day) and HIIT 2–3x/week. "
            "Reduce processed foods and increase fiber and water intake."
        ),
    },
}

BMI_CATEGORIES = [
    (0,    18.5, "Underweight"),
    (18.5, 24.9, "Normal weight"),
    (24.9, 29.9, "Overweight"),
    (29.9, float("inf"), "Obese"),
]

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def calculate_bmi(height_m: float, weight_kg: float) -> float:
    if height_m <= 0:
        raise ValueError("Height must be greater than 0.")
    return round(weight_kg / (height_m ** 2), 2)


def bmi_category(bmi: float) -> str:
    for low, high, label in BMI_CATEGORIES:
        if low <= bmi < high:
            return label
    return "Unknown"


# ──────────────────────────────────────────────
# Model (cached so it only trains once)
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():
    data = pd.read_csv("fitness_data.csv")
    data.dropna(inplace=True)

    if data["Gender"].dtype == object:
        le = LabelEncoder()
        data["Gender"] = le.fit_transform(data["Gender"])

    data["BMI"] = data["Weight_kg"] / (data["Height_m"] ** 2)

    feature_cols = ["Age", "Gender", "Height_m", "Weight_kg", "Activity_Level", "BMI"]
    X = data[feature_cols]
    y = data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy


# ──────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────

st.set_page_config(page_title="Fitness Classifier", page_icon="🏋️", layout="centered")

st.title("🏋️ Fitness Classification System")
st.caption("Enter your details below to get a personalised fitness assessment.")

# Load model
with st.spinner("Training model on fitness data…"):
    try:
        model, accuracy = load_model()
        st.success(f"Model ready — accuracy: **{accuracy:.1%}**")
    except FileNotFoundError:
        st.error("❌ `fitness_data.csv` not found. Make sure it's in the repo root.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

st.divider()

# ── Input form ──
st.subheader("Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01, format="%.2f")
    activity = st.selectbox(
        "Activity Level",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1 — Sedentary", 2: "2 — Moderately Active", 3: "3 — Very Active"}[x],
    )

with col2:
    gender = st.radio("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", horizontal=True)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0, step=0.5, format="%.1f")

st.divider()

# ── Predict button ──
if st.button("Get My Fitness Report", type="primary", use_container_width=True):
    bmi = calculate_bmi(height, weight)
    bmi_cat = bmi_category(bmi)

    features = [[age, gender, height, weight, activity, bmi]]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100

    info = LABELS[prediction]

    st.divider()
    st.subheader("📊 Your Fitness Report")

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("BMI", f"{bmi:.1f}", bmi_cat)
    m2.metric("Fitness Category", f"{info['emoji']} {info['category']}")
    m3.metric("Model Confidence", f"{confidence:.1f}%")

    # Result card
    st.markdown(
        f"""
        <div style="
            background-color: {info['color']}18;
            border-left: 5px solid {info['color']};
            border-radius: 8px;
            padding: 20px 24px;
            margin-top: 16px;
        ">
            <h3 style="color: {info['color']}; margin: 0 0 10px 0;">
                {info['emoji']} {info['category']}
            </h3>
            <p style="margin: 0; font-size: 16px; line-height: 1.6;">
                {info['recommendation']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confidence breakdown
    with st.expander("See confidence breakdown"):
        for idx, label_info in LABELS.items():
            prob = probabilities[idx] * 100
            st.write(f"{label_info['emoji']} **{label_info['category']}**")
            st.progress(int(prob), text=f"{prob:.1f}%")
