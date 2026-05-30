import streamlit as st
import numpy as np
import pickle

# Load ML Model
model = pickle.load(open("model.pkl", "rb"))

# Page Config
st.set_page_config(
    page_title="COVID Prediction",
    page_icon="🦠",
    layout="centered"
)

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

/* Animated Background */

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

/* Background Animation */

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* Glassmorphism Container */

.block-container {
    background: rgba(255,255,255,0.08);
    padding: 2rem;
    border-radius: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Main Heading */

.main-title {
    text-align: center;
    font-size: 55px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
    animation: fadeIn 1.5s ease;
}

/* Subtitle */

.sub-title {
    text-align: center;
    color: #dfe6e9;
    font-size: 20px;
    margin-bottom: 35px;
    animation: fadeIn 2s ease;
}

/* Slider Label */

label {
    color: white !important;
    font-weight: bold !important;
}

/* Slider spacing */

.stSlider {
    padding-bottom: 15px;
}

/* Button Styling */

.stButton > button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    font-size: 20px;
    font-weight: bold;
    transition: 0.4s;
}

/* Button Hover */

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg,#0072ff,#00c6ff);
}

/* Result Box */

.result-box {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin-top: 30px;
    animation: fadeIn 1s ease;
}

/* Risk Levels */

.low {
    background: rgba(46, 204, 113, 0.2);
    color: #2ecc71;
    border: 2px solid #2ecc71;
}

.medium {
    background: rgba(241, 196, 15, 0.2);
    color: #f1c40f;
    border: 2px solid #f1c40f;
}

.high {
    background: rgba(231, 76, 60, 0.2);
    color: #e74c3c;
    border: 2px solid #e74c3c;
}

/* Fade Animation */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(25px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

</style>
""", unsafe_allow_html=True)

# ======================
# HEADING
# ======================

st.markdown(
    '<div class="main-title">🦠 COVID PREDICTION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Machine Learning Based COVID Severity Predictor</div>',
    unsafe_allow_html=True
)

# ======================
# SLIDERS
# ======================

log_deaths = st.slider(
    "log_Deaths",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

log_active = st.slider(
    "log_Active",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

log_week_change = st.slider(
    "log_1 week change",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

week_increase = st.slider(
    "1 week % increase",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

deaths_100_cases = st.slider(
    "Deaths / 100 Cases",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

recovered_100_cases = st.slider(
    "Recovered / 100 Cases",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

deaths_100_recovered = st.slider(
    "Deaths / 100 Recovered",
    min_value=0.0,
    max_value=100.0,
    value=1.0,
    step=0.1
)

# ======================
# WHO REGION
# ======================

who_americas = st.slider(
    "WHO Region_Americas",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

who_eastern = st.slider(
    "WHO Region_Eastern Mediterranean",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

who_europe = st.slider(
    "WHO Region_Europe",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

who_south_east = st.slider(
    "WHO Region_South-East Asia",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

who_western = st.slider(
    "WHO Region_Western Pacific",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

# ======================
# PREDICTION BUTTON
# ======================

if st.button("Predict COVID Risk"):

    input_data = np.array([[
        log_deaths,
        log_active,
        log_week_change,
        week_increase,
        deaths_100_cases,
        recovered_100_cases,
        deaths_100_recovered,
        who_americas,
        who_eastern,
        who_europe,
        who_south_east,
        who_western
    ]])

    prediction = model.predict(input_data)

    predicted_value = prediction[0]

    # Risk Classification

    if predicted_value < 4:
        risk = "LOW RISK"
        css_class = "low"

    elif predicted_value < 5:
        risk = "MEDIUM RISK"
        css_class = "medium"

    else:
        risk = "HIGH RISK"
        css_class = "high"

    # Display Result

    st.markdown(
        f"""
        <div class="result-box {css_class}">
            Prediction Result <br><br>
            {risk}
        </div>
        """,
        unsafe_allow_html=True
    )
