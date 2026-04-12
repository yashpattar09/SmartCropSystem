# ============================================================
# SMART CROP RECOMMENDATION & PRICE PREDICTION SYSTEM
# app.py — Streamlit Dashboard (VS Code / Local)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import pickle
import os

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop System",
    page_icon="🌾",
    layout="wide"
)

# ── DARK DASHBOARD CSS ───────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #ffffff; }
    section[data-testid="stSidebar"] {
        background-color: #1a1d27;
        border-right: 1px solid #2e3250;
    }
    .card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        border: 1px solid #2e3250;
    }
    .metric-box {
        background-color: #252840;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #3a3f6e;
    }
    .metric-label { font-size: 13px; color: #9da5c9; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 700; color: #ffffff; }
    .crop-banner {
        background: linear-gradient(135deg, #1db954 0%, #0e7a36 100%);
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 16px 0;
    }
    .crop-banner h1 { font-size: 42px; margin: 0; color: white; }
    .crop-banner p  { font-size: 15px; color: #d4f5e2; margin: 6px 0 0 0; }
    .warn-banner {
        background-color: #2d2010;
        border: 1px solid #f39c12;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f39c12;
        font-size: 14px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #a0aec0;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    label { color: #c0c8e8 !important; font-size: 14px !important; }
    div.stButton > button {
        background: linear-gradient(135deg, #1db954, #0e7a36);
        color: white;
        font-size: 16px;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 0;
        width: 100%;
        cursor: pointer;
    }
    div.stButton > button:hover { opacity: 0.85; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── PATHS — UPDATE THIS TO YOUR LOCAL FOLDER ─────────────────
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models")
DATA_PATH  = os.path.join(BASE_PATH, "datasets")

# ── LOAD MODELS ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    crop_model = pickle.load(open(os.path.join(MODEL_PATH, "crop_model.pkl"), 'rb'))
    le_crop    = pickle.load(open(os.path.join(MODEL_PATH, "le_crop.pkl"),    'rb'))
    return crop_model, le_crop

@st.cache_data
def load_price_data():
    df = pd.read_csv(os.path.join(DATA_PATH, "Agriculture_price_dataset.csv"))
    df.columns = ['State', 'District', 'Market', 'Commodity',
                  'Variety', 'Grade', 'Min_Price', 'Max_Price',
                  'Modal_Price', 'Price_Date']
    df['Price_Date'] = pd.to_datetime(df['Price_Date'])
    df['State']      = df['State'].str.strip()
    df['Commodity']  = df['Commodity'].str.strip()
    df = df[df['Modal_Price'] > 0]
    return df

crop_model, le_crop = load_models()
price_df            = load_price_data()
available_crops     = [c.lower() for c in price_df['Commodity'].unique()]

# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:36px; font-weight:800; color:#1db954; margin:0;'>
        Smart Crop System
    </h1>
    <p style='color:#9da5c9; font-size:15px; margin-top:6px;'>
        AI-powered Crop Recommendation & Price Forecasting for Indian Farmers
    </p>
</div>
<hr style='border:1px solid #2e3250; margin-bottom:24px;'>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-title'>Enter Field Conditions</div>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#9da5c9; font-size:12px; margin-top:-8px;'>"
                "Default values predict Rice</p>", unsafe_allow_html=True)
    st.markdown("---")

    N           = st.number_input("Nitrogen (N)",        min_value=0,   max_value=140, value=90,    help="kg/ha")
    P           = st.number_input("Phosphorus (P)",      min_value=0,   max_value=145, value=42,    help="kg/ha")
    K           = st.number_input("Potassium (K)",       min_value=0,   max_value=205, value=43,    help="kg/ha")
    temperature = st.number_input("Temperature (C)",     min_value=0.0, max_value=50.0,value=20.8,  help="Celsius")
    humidity    = st.number_input("Humidity (%)",        min_value=0.0, max_value=100.0,value=82.0, help="Percentage")
    ph          = st.number_input("Soil pH",             min_value=0.0, max_value=14.0, value=6.5,  help="0 to 14")
    rainfall    = st.number_input("Rainfall (mm)",       min_value=0.0, max_value=300.0,value=202.9,help="Annual mm")

    st.markdown("---")
    predict_btn = st.button("Predict Crop")

# ── PREDICTION ───────────────────────────────────────────────
if predict_btn:

    input_data   = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N','P','K','temperature','humidity','ph','rainfall'])
    pred_encoded = crop_model.predict(input_data)[0]
    crop_name    = le_crop.inverse_transform([pred_encoded])[0]

    # Crop banner
    st.markdown(f"""
    <div class='crop-banner'>
        <p>Recommended Crop</p>
        <h1>{crop_name.title()}</h1>
        <p>Based on your soil and climate conditions</p>
    </div>
    """, unsafe_allow_html=True)

    # Input summary
    st.markdown("<div class='section-title'>Your Input Summary</div>",
                unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    for col, label, val, unit in zip(
        [c1,c2,c3,c4,c5,c6,c7],
        ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','Soil pH','Rainfall'],
        [N, P, K, temperature, humidity, ph, rainfall],
        ['kg/ha','kg/ha','kg/ha','C','%','','mm']
    ):
        col.markdown(f"""
        <div class='metric-box'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}
                <span style='font-size:12px; color:#9da5c9;'>{unit}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Price section
    st.markdown("<div class='section-title'>Market Price Analysis</div>",
                unsafe_allow_html=True)

    if crop_name.lower() not in available_crops:
        st.markdown(f"""
        <div class='warn-banner'>
            No price data available for <b>{crop_name.title()}</b>.<br>
            Price data is available for:
            <b>{', '.join([c.title() for c in available_crops])}</b>
        </div>""", unsafe_allow_html=True)

    else:
        crop_data               = price_df[price_df['Commodity'].str.lower() == crop_name.lower()].copy()
        crop_data['YearMonth']  = crop_data['Price_Date'].dt.to_period('M')
        monthly                 = crop_data.groupby('YearMonth')['Modal_Price'].mean().reset_index()
        monthly['YearMonth_dt'] = monthly['YearMonth'].dt.to_timestamp()
        monthly                 = monthly.sort_values('YearMonth_dt')
        monthly['Month_Num']    = np.arange(len(monthly))

        X_p = monthly[['Month_Num']]
        y_p = monthly['Modal_Price']
        reg = LinearRegression()
        reg.fit(X_p, y_p)

        last_num      = monthly['Month_Num'].max()
        future_nums   = pd.DataFrame({'Month_Num': [last_num+1, last_num+2, last_num+3]})
        future_prices = reg.predict(future_nums)
        last_date     = monthly['YearMonth_dt'].max()
        future_dates  = [last_date + pd.DateOffset(months=i) for i in [1,2,3]]
        trend         = "Increasing" if reg.coef_[0] > 0 else "Decreasing"
        trend_icon    = "+" if trend == "Increasing" else "-"

        # Price metrics
        m1,m2,m3,m4 = st.columns(4)
        for col, label, val in zip(
            [m1,m2,m3,m4],
            ['Avg Price','Lowest Price','Highest Price','Market Trend'],
            [f"Rs.{y_p.mean():.0f}", f"Rs.{y_p.min():.0f}",
             f"Rs.{y_p.max():.0f}", f"{trend_icon} {trend}"]
        ):
            col.markdown(f"""
            <div class='metric-box'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Chart
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')

        ax.plot(monthly['YearMonth_dt'], monthly['Modal_Price'],
                color='#4fc3f7', linewidth=2.5, marker='o', markersize=4,
                label='Historical Price')
        ax.plot(monthly['YearMonth_dt'], reg.predict(X_p),
                color='#ff7043', linewidth=1.5, linestyle='--', label='Trend Line')
        ax.plot(future_dates, future_prices,
                color='#66bb6a', linewidth=2.5, marker='D', markersize=8,
                linestyle='--', label='Predicted (Next 3 Months)')

        for date, price in zip(future_dates, future_prices):
            ax.annotate(f'Rs.{price:.0f}', xy=(date, price),
                        xytext=(0, 14), textcoords='offset points',
                        ha='center', fontsize=10, fontweight='bold', color='#a5d6a7')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.tick_params(colors='#9da5c9', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e3250')
        ax.set_xlabel('Month', color='#9da5c9', fontsize=11)
        ax.set_ylabel('Modal Price (Rs. / Quintal)', color='#9da5c9', fontsize=11)
        ax.set_title(f'{crop_name.title()} — Price Trend & 3-Month Forecast',
                     color='#ffffff', fontsize=14, fontweight='bold', pad=12)
        ax.legend(facecolor='#252840', edgecolor='#3a3f6e',
                  labelcolor='white', fontsize=10)
        ax.grid(True, alpha=0.15, color='#ffffff')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Forecast table
        st.markdown("<div class='section-title' style='margin-top:20px;'>"
                    "3-Month Price Forecast</div>", unsafe_allow_html=True)
        forecast_df = pd.DataFrame({
            'Month'           : [d.strftime('%B %Y') for d in future_dates],
            'Predicted Price' : [f"Rs. {p:.0f} / Quintal" for p in future_prices],
            'vs Avg'          : [f"{'Above' if future_prices[i] > y_p.mean() else 'Below'} average"
                                 for i in range(3)]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

else:
    # Landing state
    st.markdown("""
    <div class='card' style='text-align:center; padding: 60px 20px;'>
        <h2 style='color:#4a5080; font-size:28px;'>Enter your field conditions</h2>
        <p style='color:#4a5080; font-size:15px; margin-top:8px;'>
            Fill in the soil and climate values in the sidebar,<br>
            then click <b style='color:#1db954;'>Predict Crop</b> to get your
            recommendation and price forecast.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Price Data Available For</div>",
                unsafe_allow_html=True)
    icons = {'wheat':'🌾','rice':'🍚','potato':'🥔','onion':'🧅','tomato':'🍅'}
    cols  = st.columns(5)
    for col, crop in zip(cols, available_crops):
        col.markdown(f"""
        <div class='metric-box'>
            <div style='font-size:28px;'>{icons.get(crop, '')}</div>
            <div class='metric-label' style='margin-top:6px;'>{crop.title()}</div>
        </div>""", unsafe_allow_html=True)
