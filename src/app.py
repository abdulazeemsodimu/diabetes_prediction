import streamlit as st
from predict import prediction


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)


# --------------------------------------------------
# Title and introduction
# --------------------------------------------------

st.title("🩺 Diabetes Risk Predictor")

st.write(
    """
    Enter the patient's information below to obtain a prediction
    from the trained machine learning model.
    """
)

st.info(
    """
    This application is for educational and demonstration purposes.
    It is not intended to provide a medical diagnosis.
    """
)


# --------------------------------------------------
# Patient information form
# --------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Patient Information")

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0,
        help="Plasma glucose concentration."
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0,
        help="Diastolic blood pressure."
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        format="%.1f"
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.01,
        format="%.2f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1
    )

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:

        probability, pred = prediction(input_data)

        probability_percentage = probability * 100

        st.divider()

        st.subheader("Prediction Result")

        if pred == 1:

            st.warning(
                "The model predicts a higher likelihood of diabetes."
            )

        else:

            st.success(
                "The model predicts a lower likelihood of diabetes."
            )

        st.metric(
            "Estimated probability of diabetes",
            f"{probability_percentage:.2f}%"
        )

        st.progress(probability)

        st.caption(
            """
            The probability shown is the model's estimated probability
            for the positive class. It should not be interpreted as a
            medical diagnosis.
            """
        )

    except Exception as e:

        st.error(
            f"An error occurred while making the prediction: {e}"
        )

