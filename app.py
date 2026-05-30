import streamlit as st
import numpy as np
import joblib
import os

# ======================
# PAGE CONFIG
# ======================

st.set_page_config(
    page_title="COVID Prediction",
    page_icon="🦠",
    layout="centered"
)

# ======================
# LOAD MODEL SAFELY
# ======================

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("model.pkl file not found. Upload model file to repository.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ======================
# CUSTOM CSS
# ======================

st.markdown("""
<style>

.stApp{
background:linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c92d2);
background-size:400% 400%;
animation:gradient 12s ease infinite;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.block-container{
background:rgba(255,255,255,0.08);
padding:2rem;
border-radius:20px;
backdrop-filter:blur(10px);
}

.main-title{
text-align:center;
font-size:50px;
font-weight:bold;
color:white;
}

.sub-title{
text-align:center;
font-size:20px;
color:#ddd;
margin-bottom:20px;
}

.result-box{
padding:25px;
border-radius:15px;
font-size:28px;
font-weight:bold;
text-align:center;
margin-top:20px;
}

.low{
background:rgba(46,204,113,.2);
color:#2ecc71;
}

.medium{
background:rgba(241,196,15,.2);
color:#f1c40f;
}

.high{
background:rgba(231,76,60,.2);
color:#e74c3c;
}

</style>
""", unsafe_allow_html=True)

# ======================
# TITLE
# ======================

st.markdown(
'<div class="main-title">🦠 COVID Prediction</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="sub-title">Machine Learning Based COVID Severity Predictor</div>',
unsafe_allow_html=True
)

# ======================
# INPUTS
# ======================

log_deaths=st.slider("log_Deaths",0.0,20.0,1.0)

log_active=st.slider("log_Active",0.0,20.0,1.0)

log_week_change=st.slider("log_1 Week Change",0.0,20.0,1.0)

week_increase=st.slider("1 Week % Increase",0.0,100.0,1.0)

deaths_100_cases=st.slider("Deaths /100 Cases",0.0,100.0,1.0)

recovered_100_cases=st.slider("Recovered /100 Cases",0.0,100.0,1.0)

deaths_100_recovered=st.slider("Deaths /100 Recovered",0.0,100.0,1.0)

who_americas=st.slider("WHO Region Americas",0,10,0)

who_eastern=st.slider("WHO Region Eastern",0,10,0)

who_europe=st.slider("WHO Region Europe",0,10,0)

who_south_east=st.slider("WHO Region South-East Asia",0,10,0)

who_western=st.slider("WHO Region Western Pacific",0,10,0)

# ======================
# PREDICT BUTTON
# ======================

if st.button("Predict COVID Risk"):

    data=np.array([[
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

    try:

        prediction=model.predict(data)

        value=prediction[0]

        if value<4:

            risk="LOW RISK"
            css="low"

        elif value<5:

            risk="MEDIUM RISK"
            css="medium"

        else:

            risk="HIGH RISK"
            css="high"

        st.markdown(
        f"""
        <div class="result-box {css}">
        {risk}
        </div>
        """,
        unsafe_allow_html=True
        )

    except Exception as e:

        st.error(f"Prediction failed : {e}")
