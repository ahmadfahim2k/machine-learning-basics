import streamlit as st
from prediction_helper import predict

# --------------------------
# Configuration dictionary
# --------------------------
FIELDS = {
    "categorical": {
        "gender": ['Male', 'Female'],
        "region": ['Northeast', 'Northwest', 'Southeast', 'Southwest'],
        "marital_status": ['Unmarried', 'Married'],
        "bmi_category": ['Overweight', 'Underweight', 'Normal', 'Obesity'],
        "smoking_status": [
            'Regular', 'No Smoking', 'Occasional'
        ],
        "employment_status": ['Self-Employed', 'Freelancer', 'Salaried'],
        "income_level": ['> 40L', '<10L', '10L - 25L', '25L - 40L'],
        "medical_history": [
            'High blood pressure', 'No Disease', 'Diabetes & High blood pressure',
            'Diabetes & Heart disease', 'Diabetes', 'Diabetes & Thyroid',
            'Heart disease', 'Thyroid', 'High blood pressure & Heart disease'
        ],
        "insurance_plan": ['Silver', 'Bronze', 'Gold']
    },
    "numerical": {
        "age": {"min": 18, "max": 100, "step": 1},
        "income_lakhs": {"min": 0, "max": 200, "step": 1},
        "number_of_dependants": {"min": 0, "max": 20, "step": 1},
        "genetical_risk": {"min": 0, "max": 5, "step": 1}
    }
}

# --------------------------
# Streamlit UI
# --------------------------
st.title("ML Model Tester")

st.subheader("Enter Input Data:")

user_input = {}

# Field order (income_lakhs before income_level)
field_order = [
    "age", "number_of_dependants", "income_lakhs", "genetical_risk",
    "insurance_plan", "employment_status",  "gender", "marital_status", 
    "bmi_category", "smoking_status", "region", "medical_history", 
    "income_level"
]

# Display fields in 4 columns per row
for i in range(0, len(field_order), 3):
    cols = st.columns(3)
    for col_index, field in enumerate(field_order[i:i+3]):
        with cols[col_index]:
            label = field.replace('_', ' ').title()

            # Numerical input
            if field in FIELDS["numerical"]:
                cfg = FIELDS["numerical"][field]
                user_input[field] = st.number_input(
                    label,
                    min_value=cfg["min"],
                    max_value=cfg["max"],
                    step=cfg["step"]
                )

            # Income level (disabled, auto-calculated)
            elif field == "income_level":
                income = user_input.get("income_lakhs", 0)
                if income < 10:
                    auto_val = "<10L"
                elif 10 <= income <= 25:
                    auto_val = "10L - 25L"
                elif 25 < income <= 40:
                    auto_val = "25L - 40L"
                else:
                    auto_val = "> 40L"

                user_input[field] = st.selectbox(
                    label,
                    FIELDS["categorical"]["income_level"],
                    index=FIELDS["categorical"]["income_level"].index(auto_val),
                    disabled=True
                )

            # Regular categorical input
            else:
                user_input[field] = st.selectbox(label, FIELDS["categorical"].get(field, []))

# Predict button
if st.button("Predict"):
    prediction = predict(user_input)
    st.success(f"Predicted Premium : {prediction}")