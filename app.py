import streamlit as st
import pandas as pd
import joblib


st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(145deg, #1e2229, #292e36);
        border-radius: 12px;
        padding: 10px 20px;
        border: 1px solid #2c313a;
        color: #fff;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(145deg, #292e36, #1e2229);
        transform: scale(1.05);
    }
    .stSelectbox, .stNumberInput {
        background-color: #1e2229 !important;
        color: white !important;
        border-radius: 10px;
        padding: 8px;
    }
    .css-1aumxhk {
        background-color: #1e2229;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 5px 5px 10px #12161d, -5px -5px 10px #2c313a;
    }
    .css-15zrgzn {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


model_path = "best_model_xgboost.pkl"  # Adjust if inside a folder, e.g., "models/best_model_xgboost.pkl"
model = joblib.load(model_path)


feature_names = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape',
                 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
                 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                 '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
                 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
                 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
                 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']


categorical_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'BldgType', 
                    'HouseStyle', 'RoofStyle', 'GarageType', 'PavedDrive']


categorical_features = {
    "MSZoning": ["RL", "RM", "C (all)", "FV", "RH"],
    "Street": ["Paved", "Gravel"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandContour": ["Lvl", "Bnk", "HLS", "Low"],
    "Utilities": ["AllPub", "NoSeWa"],
    "BldgType": ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
    "HouseStyle": ["1Story", "2Story", "1.5Fin", "1.5Unf", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"],
    "RoofStyle": ["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"],
    "GarageType": ["Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types"],
    "PavedDrive": ["Y", "N", "P"]
}


st.sidebar.header("🏡 House Features")
input_data = {}


for feature in feature_names[:15]:  
    if feature in categorical_features:
        value = st.sidebar.selectbox(f"{feature}:", categorical_features[feature])
    else:
        value = st.sidebar.number_input(f"{feature}:", min_value=0, value=0, step=1)
    
    input_data[feature] = value


input_df = pd.DataFrame([input_data])


for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].astype('category').cat.codes


model_features = model.get_booster().feature_names 
input_df = input_df.reindex(columns=model_features, fill_value=0)  


if st.button("🔮 Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"🏡 **Estimated House Price: ${prediction:,.2f}**")


st.markdown("""
    <br><br>
    <div style="text-align: center; font-size: 14px;">
        Built with ❤️ by Harshit Rai | AI-Powered House Price Predictor
    </div>
    """, unsafe_allow_html=True)
