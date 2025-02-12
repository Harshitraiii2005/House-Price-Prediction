import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# Load trained model, scaler, and encoder
model = joblib.load("house_price_xgboost.pkl")
scaler = joblib.load("scaler.pkl")  
encoder = joblib.load("encoder.pkl")

# Load expected feature names from training
expected_features = list(scaler.feature_names_in_)

# Streamlit UI Configuration
st.set_page_config(page_title="🏡 House Price Prediction", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="🏠 House Price Predictor",
        options=["Home", "Predict"],
        icons=["house", "graph-up"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
# Apply custom CSS for background GIF
st.markdown(
    """
    <style>
        body {
            background: url("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDg4dzR6Nzcyb2hwamUxeGhyN2RoMWVmNGpoaXh4cjl3bnY2OG94OCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/x4O0fjpQfoBZS/giphy.gif") no-repeat center center fixed;
            background-size: cover;
        }
        .stApp {
            background: rgba(0, 0, 0, 0.6);
            padding: 20px;
            border-radius: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if selected == "Home":
    st.title("🏡 AI-Powered House Price Prediction")
    st.write("Get an accurate estimate of your house value using Machine Learning!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image("https://source.unsplash.com/featured/?luxury,house", use_column_width=True)
        st.write("💡 *Enter your property details to predict its price instantly!* 🏠")
        st.write("🔹 **Advanced AI Model** trained with real estate data")
        st.write("🔹 **Fast & Accurate** predictions based on key features")
        st.write("🔹 **User-Friendly** Interface for easy inputs")

    with col2:
        st.image("https://media3.giphy.com/media/W1fFapmqgqEf8RJ9TQ/giphy.gif", use_column_width=True)

# Prediction Page
elif selected == "Predict":
    st.title("📊 Predict House Price")
    st.write("Fill in the details below and get an estimated price!")

    col1, col2 = st.columns(2)

    with col1:
        lot_area = st.number_input("🏡 Lot Area (sq ft)", min_value=500, format="%d")
        bedrooms = st.number_input("🛏 Number of Bedrooms", min_value=1, format="%d")
        bathrooms = st.number_input("🛁 Number of Bathrooms", min_value=1, format="%d")
        overall_quality = st.slider("🏆 Overall Quality (1-10)", 1, 10, 5)
        year_built = st.number_input("📅 Year Built", min_value=1900, max_value=2025, format="%d")
        total_bsmt_sf = st.number_input("🏗 Total Basement Area (sq ft)", min_value=0, format="%d")
        gr_liv_area = st.number_input("🏠 Above Ground Living Area (sq ft)", min_value=0, format="%d")

    with col2:
        garage_cars = st.number_input("🚗 Garage Cars", min_value=0, max_value=5, format="%d")
        garage_area = st.number_input("🚘 Garage Area (sq ft)", min_value=0, format="%d")
        wood_deck_sf = st.number_input("🌳 Wood Deck Area (sq ft)", min_value=0, format="%d")
        open_porch_sf = st.number_input("🏡 Open Porch Area (sq ft)", min_value=0, format="%d")
        pool_area = st.number_input("🏊 Pool Area (sq ft)", min_value=0, format="%d")
        fireplace = st.selectbox("🔥 Has Fireplace?", ["Yes", "No"])
        neighborhood = st.selectbox("📍 Select Neighborhood", encoder.categories_[0])  
        house_style = st.selectbox("🏠 Select House Style", encoder.categories_[1])

    if st.button("💰 Predict House Price", use_container_width=True):
        # Encode categorical data
        cat_input = pd.DataFrame([[neighborhood, house_style]], columns=["Neighborhood", "HouseStyle"])
        cat_encoded = encoder.transform(cat_input)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["Neighborhood", "HouseStyle"]))

        # Encode fireplace manually
        fireplace_mapping = {"No": 0, "Yes": 1}
        fireplace_encoded = fireplace_mapping[fireplace]

        # Prepare numerical features
        num_features_dict = {
            "LotArea": lot_area,
            "BedroomAbvGr": bedrooms,
            "FullBath": bathrooms,
            "OverallQual": overall_quality,
            "YearBuilt": year_built,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "Fireplaces": fireplace_encoded,
            "WoodDeckSF": wood_deck_sf,
            "OpenPorchSF": open_porch_sf,
            "PoolArea": pool_area,
            "GrLivArea": gr_liv_area,
            "GarageArea": garage_area,
        }

        num_features_df = pd.DataFrame([num_features_dict])

        # Combine numerical and encoded categorical features
        final_features = pd.concat([num_features_df, cat_encoded_df], axis=1)

        # Add missing features with default value 0
        for feat in expected_features:
            if feat not in final_features.columns:
                final_features[feat] = 0  

        # Drop unnecessary features not in expected_features
        final_features = final_features[expected_features]

        # ✅ Scale features
        features_scaled = scaler.transform(final_features)

        # 🎯 Predict
        prediction = model.predict(features_scaled)

        st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")
        st.balloons()

# Footer
st.markdown(
    """
    <hr style="border: 1px solid #ccc; margin-top: 40px;">
    <div style="text-align: center; font-size: 16px;">
        <p>🚀 Built with ❤️ by <b>Harshit Rai</b> using <b>Streamlit</b></p>
        <p>🔗 Connect with me on <a href="https://github.com/Harshitraiii2005" target="_blank" style="color: #FF5733; text-decoration: none;"><b>GitHub</b></a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
