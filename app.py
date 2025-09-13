import streamlit as st
import joblib
import numpy as np
loaded_model = joblib.load("ckd_final_model.pkl")

st.title("🩺 Chronic Kidney Disease Prediction App")
st.write("Fill in the patient details below to predict CKD (Chronic Kidney Disease).")

# ----------------------------
# 1. Define Inputs
# ----------------------------

# Numeric inputs
age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=80)
sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
al = st.slider("Albumin (0–5)", 0, 5, 0)
su = st.slider("Sugar (0–5)", 0, 5, 0)
bgr = st.number_input("Blood Glucose Random (mg/dl)", min_value=50, max_value=500, value=120)
bu = st.number_input("Blood Urea (mg/dl)", min_value=0, max_value=400, value=40)
sc = st.number_input("Serum Creatinine (mg/dl)", min_value=0.0, max_value=20.0, value=1.2)
sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=200, value=135)
pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=10.0, value=4.5)
hemo = st.number_input("Hemoglobin (g/dl)", min_value=3.0, max_value=20.0, value=13.0)
pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=60, value=40)
wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=2000, max_value=20000, value=8000)
rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=2.0, max_value=8.0, value=4.5)

# Categorical inputs
rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
ba = st.selectbox("Bacteria", ["notpresent", "present"])
htn = st.selectbox("Hypertension", ["no", "yes"])
dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])
appet = st.selectbox("Appetite", ["good", "poor"])
pe = st.selectbox("Pedal Edema", ["no", "yes"])
ane = st.selectbox("Anemia", ["no", "yes"])

# ----------------------------
# 2. Encode Categorical Inputs
# ----------------------------
def encode_binary(val, positive="yes", negative="no"):
    return 1 if val == positive else 0

def encode_presence(val):
    return 1 if val == "present" else 0

def encode_normal(val):
    return 1 if val == "abnormal" else 0

def encode_appetite(val):
    return 1 if val == "poor" else 0

rbc_val = encode_normal(rbc)
pc_val = encode_normal(pc)
pcc_val = encode_presence(pcc)
ba_val = encode_presence(ba)
htn_val = encode_binary(htn)
dm_val = encode_binary(dm)
cad_val = encode_binary(cad)
appet_val = encode_appetite(appet)
pe_val = encode_binary(pe)
ane_val = encode_binary(ane)

# ----------------------------
# 3. Combine Features
# ----------------------------
features = np.array([[age, bp, sg, al, su, bgr, bu, sc, sod, pot,
                      hemo, pcv, wc, rc,
                      rbc_val, pc_val, pcc_val, ba_val,
                      htn_val, dm_val, cad_val, appet_val, pe_val, ane_val]])

# ----------------------------
# 4. Prediction
# ----------------------------
if st.button("🔍 Predict CKD"):
    prediction = loaded_model.predict(features)
    if prediction[0] == 1:
        st.error("⚠️ The model predicts: **CKD (Chronic Kidney Disease)**")
    else:
        st.success("✅ The model predicts: **Non-CKD**")

