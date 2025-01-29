import numpy as np
# import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("mhp.csv")  # Ensure the dataset path is correct
model = joblib.load("House_price_prediction.pkl")  # Load the trained model

# Preprocess the dataset
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['bhk'] = df['bhk'].astype(int)
df['area'] = pd.to_numeric(df['area'], errors='coerce')

categorical_cols = ['type', 'region', 'status', 'age']
label_encoders = {}

# Initialize LabelEncoders for categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("Unknown")  # Handle nulls
    df[col] = df[col].astype(str)
    le.fit(df[col])
    label_encoders[col] = le

# Function to encode categorical input values
def encode_input(value, column):
    try:
        return label_encoders[column].transform([value])[0]
    except ValueError:
        st.warning(f"Warning: The value '{value}' for {column} is unseen. Using default encoding (0).")
        return 0

# Streamlit UI

# Custom CSS to add a background image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.pexels.com/photos/3736687/pexels-photo-3736687.jpeg?auto=compress&cs=tinysrgb&w=600");
    background-size: fill;
    background-repeat: repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
</style>
"""

# Add the CSS to the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)

# Your Streamlit App Code
st.title("🏠 House Price Prediction App")
st.markdown("### Enter house details below:")

# Collect user inputs
bhk = st.number_input("**Number of BHK**:", min_value=1, max_value=10, value=3, step=1)
house_type = st.selectbox("**House Type**:", options=sorted(df['type'].unique()))
region = st.selectbox("**Region**:", options=sorted(df['region'].unique()))
status = st.selectbox("**Status**:", options=sorted(df['status'].unique()))
age = st.selectbox("**Age**:", options=sorted(df['age'].unique()))
area = st.number_input("**House Area (in sqft)**:", min_value=100.0, value=290.0, step=10.0)

# Price range inputs
st.markdown("### Price Range:")
price_units = st.radio("**Select price unit**:", options=["**Lakh (L)**", "**Crore (Cr)**"])
min_price_value = st.number_input("**Minimum Price**:", min_value=0.0, value=100000.0, step=10000.0)
max_price_value = st.number_input("**Maximum Price**:", min_value=0.0, value=1000000.0, step=10000.0)
area_range = 100.0  # Adjustable range
area_min = max(0, area - area_range)
area_max = area + area_range
# Filter houses based on inputs
filtered_houses = df[
    (df['price'] >= min_price_value) &
    (df['price'] <= max_price_value) &
    (df['bhk'] == bhk) &
    (df['area'] >= area_min) &
    (df['area'] <= area_max)  # Allow some range around input
]

if region.strip():
    filtered_houses = filtered_houses[filtered_houses['region'].str.lower() == region.strip().lower()]
if house_type.strip():
    filtered_houses = filtered_houses[filtered_houses['type'].str.lower() == house_type.strip().lower()]
if status.strip():
    filtered_houses = filtered_houses[filtered_houses['status'].str.lower() == status.strip().lower()]
if age.strip():
    filtered_houses = filtered_houses[filtered_houses['age'].str.lower() == age.strip().lower()]

# Display results
if filtered_houses.empty:
    st.warning("No houses found matching the criteria.")
else:
    st.write(f"🏘 Number of houses available: {len(filtered_houses)}")
    st.dataframe(filtered_houses)

    # Predict price for the first house in the filtered list
    if st.checkbox("Predict price for the first matching house"):
        house_features = filtered_houses.iloc[0][['bhk', 'type', 'area', 'region', 'status', 'age']]

        # Encode categorical columns
        try:
            house_features['type'] = encode_input(house_features['type'], 'type')
            house_features['region'] = encode_input(house_features['region'], 'region')
            house_features['status'] = encode_input(house_features['status'], 'status')
            house_features['age'] = encode_input(house_features['age'], 'age')
        except KeyError as e:
            st.error(f"Error encoding inputs: {e}")
            st.stop()

        # Convert to model's expected format
        house_features = house_features.values.reshape(1, -1)

        # Predict
        try:
            predicted_price = model.predict(house_features)
            st.success(f"Predicted Price: ₹ {predicted_price[0]:,.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
