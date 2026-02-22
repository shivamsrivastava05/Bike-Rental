# import streamlit as st
# import pickle
# import numpy as np

# # Load trained model
# model = pickle.load(open("bike_model.pkl", "rb"))

# st.title("🚲 Bike Rental Demand Prediction")

# st.subheader("Enter Input Features")

# # Categorical Inputs
# season = st.selectbox("Season (1:Spring, 2:Summer, 3:Fall, 4:Winter)", [1,2,3,4])
# holiday = st.selectbox("Holiday (0:No, 1:Yes)", [0,1])
# workingday = st.selectbox("Working Day (0:No, 1:Yes)", [0,1])
# weather = st.selectbox("Weather (1:Clear, 2:Mist, 3:Light Rain/Snow, 4:Heavy Rain/Snow)", [1,2,3,4])
# yr = st.selectbox("Year (0:2025, 1:2026)", [0,1])
# month = st.slider("Month", 1, 12)
# hour = st.slider("Hour", 0, 23)

# # Numerical Inputs
# temp = st.number_input("Temperature (Normalized)", value=0.5)
# atemp = st.number_input("Feels Like Temperature (Normalized)", value=0.5)
# humidity = st.number_input("Humidity (Normalized)", value=0.5)
# windspeed = st.number_input("Wind Speed (Normalized)", value=0.2)

# # Arrange features EXACTLY in same order as training
# features = np.array([[season, holiday, workingday, weather,temp, atemp, humidity, windspeed,yr, month, hour]])

# # Prediction
# if st.button("Predict"):
#     prediction = model.predict(features)
#     st.success(f"Predicted Bike Rentals: {int(prediction[0])}")
import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("bike_model.pkl", "rb"))

st.title("🚲 Bike Rental Demand Prediction")

st.markdown("### Enter Bike Rental Conditions")

# ---- Input Fields ----
season = st.selectbox("Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)", [1,2,3,4])
yr = st.selectbox("Year (0=2025, 1=2026)", [0,1])
month = st.slider("Month", 1, 12)
holiday = st.selectbox("Holiday (0=No, 1=Yes)", [0,1])
workingday = st.selectbox("Working Day (0=No, 1=Yes)", [0,1])
weather = st.selectbox("Weather (1-4)", [1,2,3,4])

temp = st.number_input("Temperature (Normalized)", min_value=0.0)
atemp = st.number_input("Feeling Temperature (Normalized)", min_value=0.0)
humidity = st.number_input("Humidity", min_value=0.0)
windspeed = st.number_input("Wind Speed", min_value=0.0)

hour = st.slider("Hour of the Day", 0, 23)

# ---- Arrange Features (MUST match training order) ----
features = np.array([[season, yr, month, holiday,
                      workingday, weather,
                      temp, atemp,
                      humidity, windspeed, hour]])

# ---- Prediction ----
if st.button("Predict"):
    prediction_log = model.predict(features)

    # If you trained on log(count)
    prediction = np.exp(prediction_log)

    st.subheader("📊 Predicted Bike Rentals:")
    st.success(int(prediction[0]))

# ---- Debug Info (optional – remove later) ----
# st.write("Model expects:", model.n_features_in_)
# st.write("Feature shape:", features.shape)