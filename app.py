"""
Fitness Classification System
Uses Random Forest ML to predict fitness category and provide recommendations.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

LABELS = {
    0: {
        "category": "Underfit",
        "emoji": "💪",
        "recommendation": (
            "Increase daily calorie intake with protein-rich foods "
            "(eggs, chicken, legumes). Focus on strength training 3–4x/week "
            "and ensure 7–9 hours of sleep for muscle recovery."
        ),
    },
    1: {
        "category": "Fit",
        "emoji": "✅",
        "recommendation": (
            "Maintain your balanced diet and consistent exercise routine. "
            "Mix cardio and strength training. Stay hydrated and monitor "
            "progress monthly to keep on track."
        ),
    },
    2: {
        "category": "Overfit",
        "emoji": "🔥",
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

ACTIVITY_LABELS = {1: "Sedentary", 2: "Moderately Active", 3: "Very Active"}


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def calculate_bmi(height_m: float, weight_kg: float) -> float:
    """Return BMI rounded to 2 decimal places."""
    if height_m <= 0:
        raise ValueError("Height must be greater than 0.")
    return round(weight_kg / (height_m ** 2), 2)


def bmi_category(bmi: float) -> str:
    """Return a human-readable BMI category string."""
    for low, high, label in BMI_CATEGORIES:
        if low <= bmi < high:
            return label
    return "Unknown"


def get_int_input(prompt: str, valid_range: range | list | None = None) -> int:
    """Prompt the user for an integer, validating against an optional range."""
    while True:
        try:
            value = int(input(prompt).strip())
            if valid_range is not None and value not in valid_range:
                print(f"  ⚠  Please enter one of: {list(valid_range)}")
                continue
            return value
        except ValueError:
            print("  ⚠  Invalid input — please enter a whole number.")


def get_float_input(prompt: str, min_val: float = 0.0, max_val: float = 1e9) -> float:
    """Prompt the user for a float within an optional [min, max] range."""
    while True:
        try:
            value = float(input(prompt).strip())
            if not (min_val < value <= max_val):
                print(f"  ⚠  Please enter a value between {min_val} and {max_val}.")
                continue
            return value
        except ValueError:
            print("  ⚠  Invalid input — please enter a number.")


# ──────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────

def load_and_prepare_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, drop nulls, encode categoricals, and engineer the BMI feature.
    Returns (X, y).
    """
    data = pd.read_csv(filepath)

    # Drop rows with missing values
    initial_rows = len(data)
    data.dropna(inplace=True)
    dropped = initial_rows - len(data)
    if dropped:
        print(f"  ℹ  Dropped {dropped} rows with missing values.")

    # Encode Gender if it's a string (e.g., 'Male'/'Female')
    if data["Gender"].dtype == object:
        le = LabelEncoder()
        data["Gender"] = le.fit_transform(data["Gender"])

    # Engineer BMI as an additional feature
    data["BMI"] = data.apply(
        lambda row: calculate_bmi(row["Height_m"], row["Weight_kg"]), axis=1
    )

    feature_cols = ["Age", "Gender", "Height_m", "Weight_kg", "Activity_Level", "BMI"]
    X = data[feature_cols]
    y = data["Label"]

    return X, y


# ──────────────────────────────────────────────
# Model Training
# ──────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    """
    Split data, train a Random Forest, and print evaluation metrics.
    Returns the trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'─'*45}")
    print(f"  Model Accuracy : {accuracy:.2%}")
    print(f"{'─'*45}")
    print(classification_report(y_test, y_pred, target_names=["Underfit", "Fit", "Overfit"]))

    return model


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def collect_user_input() -> dict:
    """Interactively collect user measurements and return them as a dict."""
    print("\n" + "═" * 45)
    print("   🏋️  FITNESS ASSESSMENT — Enter Your Details")
    print("═" * 45)

    age      = get_int_input("  Age              : ", range(1, 121))
    gender   = get_int_input("  Gender (0=F, 1=M): ", [0, 1])
    height   = get_float_input("  Height (m)       : ", 0.5, 2.5)
    weight   = get_float_input("  Weight (kg)      : ", 10.0, 500.0)
    activity = get_int_input(
        "  Activity Level\n  (1=Sedentary, 2=Moderate, 3=Active): ",
        [1, 2, 3]
    )

    return {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "activity": activity,
    }


def predict_and_display(model: RandomForestClassifier, user: dict) -> None:
    """Run prediction and print a formatted results report."""
    bmi = calculate_bmi(user["height"], user["weight"])
    bmi_cat = bmi_category(bmi)
    activity_label = ACTIVITY_LABELS.get(user["activity"], "Unknown")

    features = [[
        user["age"], user["gender"], user["height"],
        user["weight"], user["activity"], bmi
    ]]

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100

    info = LABELS[prediction]

    print("\n" + "═" * 45)
    print("   📊  YOUR FITNESS REPORT")
    print("═" * 45)
    print(f"  BMI              : {bmi:.2f}  ({bmi_cat})")
    print(f"  Activity Level   : {activity_label}")
    print(f"  Fitness Category : {info['emoji']}  {info['category']}")
    print(f"  Model Confidence : {confidence:.1f}%")
    print(f"\n  📋 Recommendation:")
    # Word-wrap recommendation at ~50 chars
    words = info["recommendation"].split()
    line, lines = "", []
    for word in words:
        if len(line) + len(word) + 1 > 50:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    for l in lines:
        print(f"     {l}")
    print("═" * 45 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("\n🔄  Loading data and training model…")
    X, y = load_and_prepare_data("fitness_data.csv")
    model = train_model(X, y)

    while True:
        user = collect_user_input()
        predict_and_display(model, user)

        again = input("  Run another assessment? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye — stay fit! 👋\n")
            break


if __name__ == "__main__":
    main()
