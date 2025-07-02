import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== 1. إعداد الصفحة ==========
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction")
st.subheader("🧮 Predict Selling Price")
st.write("Enter vehicle specifications to predict the selling price:")

# ========== 2. تحميل النموذج ==========
@st.cache_resource
def load_model():
    return joblib.load("car_price_model_compressed.pkl")

model = load_model()

# ========== 3. القيم الممكنة يدويًا ==========
fuel_options = ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric']
seller_type_options = ['Individual', 'Dealer', 'Trustmark Dealer']
transmission_options = ['Manual', 'Automatic']
owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
brand_options = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Audi', 'Mercedes-Benz']  # عدل حسب بياناتك
model_options = ['Swift VDI', 'i20 Sportz', 'City VX', 'Fortuner 2.8', 'EcoSport Titanium']  # عدل حسب بياناتك

# ========== 4. واجهة الإدخال ==========
col1, col2 = st.columns(2)

with col1:
    km_driven = st.number_input("KM Driven", min_value=0, max_value=1000000, step=5000)
    fuel = st.selectbox("Fuel Type", fuel_options)
    seller_type = st.selectbox("Seller Type", seller_type_options)

with col2:
    transmission = st.selectbox("Transmission", transmission_options)
    owner = st.selectbox("Owner Type", owner_options)
    car_age = st.slider("Car Age", 0, 25, 5)

# اختيارات عامة للماركة والموديل
brand = st.selectbox("Brand", brand_options)
model_text = st.selectbox("Model", model_options)

# ========== 5. التنبؤ بالسعر ==========
if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "car_age": car_age,
        "brand": brand,
        "model": model_text
    }])

    # إجراء التنبؤ
    prediction = model.predict(input_data)
    predicted_price = np.expm1(prediction[0])  # عكس log1p

    st.success(f"🚘 Estimated Selling Price: ₹ {predicted_price:,.0f}")
    st.caption("Developed by Ali Ahmed Zaki")
