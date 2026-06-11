import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_pipeline import load_and_sync_data, HRTAGoldScraper
from forecaster import GoldForecaster
from margin_corrector import calculate_historical_spreads, apply_spread_correction

# Set page configuration for premium feel
st.set_page_config(
    page_title="HRTA Gold Forecaster",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Gold & Dark Theme Aesthetics
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Title Header with Gold Gradient */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        border: 1px solid rgba(201, 150, 44, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .main-title {
        background: linear-gradient(90deg, #C9962C 0%, #F3E5AB 50%, #C9962C 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: 1px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 5px;
        font-weight: 300;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(201, 150, 44, 0.4);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-value.gold {
        color: #C9962C;
    }
    .metric-value.blue {
        color: #185FA5;
    }
    .metric-value.orange {
        color: #D85A30;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .delta-up {
        color: #10b981;
    }
    .delta-down {
        color: #ef4444;
    }
    
    /* Footer Credit */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & CACHED DATA LOADING -----------------
if 'data_trigger' not in st.session_state:
    st.session_state.data_trigger = 0

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(trigger):
    # Returns synchronized dataframe
    return load_and_sync_data(csv_path="data/gold_price_history.csv")

with st.spinner("Synchronizing spot and live retail gold prices..."):
    db_df = get_data(st.session_state.data_trigger)

# Parse components
db_df['Date'] = pd.to_datetime(db_df['Date'])
latest_row = db_df.iloc[-1]
prev_row = db_df.iloc[-2] if len(db_df) > 1 else latest_row

# Calculate historical spreads
spreads = calculate_historical_spreads(db_df)

# ----------------- SIDEBAR CONFIGURATIONS -----------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 20px;'>
        <h2 style='color: #C9962C; margin: 0;'>⚙️ Controls</h2>
        <p style='color: #64748b; font-size: 0.85rem;'>Configure Forecaster & Model Engines</p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("### 📅 Plot Range")
lookback_options = {
    "Last 30 Days": 30,
    "Last 90 Days": 90,
    "Last 180 Days": 180,
    "Last 365 Days": 365,
    "All Available History": len(db_df)
}
selected_range = st.sidebar.selectbox("Lookback Window", list(lookback_options.keys()), index=1)
lookback_days = lookback_options[selected_range]

st.sidebar.markdown("### ⚖️ Spread Model")
spread_mode = st.sidebar.radio(
    "Margin Correction Spread",
    ["Use Latest Scraped Daily Spread", "Use 30-Day Smoothed Average"],
    index=0
)
use_latest_spread = (spread_mode == "Use Latest Scraped Daily Spread")

st.sidebar.markdown("### 🤖 SARIMAX Settings")
p_order = st.sidebar.slider("Autoregressive Order (p)", 0, 5, 1)
d_order = st.sidebar.slider("Differencing Order (d)", 0, 2, 1)
q_order = st.sidebar.slider("Moving Average Order (q)", 0, 5, 1)

# Action Buttons
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Sync Live Data Now", use_container_width=True):
    st.cache_data.clear()
    st.session_state.data_trigger += 1
    st.rerun()

# ----------------- HEADER & BANNER -----------------
st.markdown(
    f"""
    <div class="header-container">
        <h1 class="main-title">🪙 HRTA GOLD FORECASTER</h1>
        <div class="subtitle">
            Dual-Engine Retail Gold Forecasting: SARIMAX (Spot) + Spread Corrector (Physical) | 
            Database Synced: <b>{latest_row['Date'].strftime('%Y-%m-%d')}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- LIVE RETAIL METRICS -----------------
# Calculate changes
retail_change = latest_row['Retail_Price'] - prev_row['Retail_Price']
retail_pct = (retail_change / prev_row['Retail_Price']) * 100 if prev_row['Retail_Price'] > 0 else 0
spot_change = latest_row['Spot_IDR_Gram'] - prev_row['Spot_IDR_Gram']
spot_pct = (spot_change / prev_row['Spot_IDR_Gram']) * 100 if prev_row['Spot_IDR_Gram'] > 0 else 0

active_spread = spreads['latest_retail_spread'] if use_latest_spread else spreads['avg_retail_spread']
active_spread_pct = active_spread * 100
spread_diff = latest_row['Retail_Price'] - latest_row['Spot_IDR_Gram']

col1, col2, col3 = st.columns(3)

with col1:
    delta_class = "delta-up" if retail_change >= 0 else "delta-down"
    sign = "+" if retail_change >= 0 else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">🪙 Physical Retail Price (Beli)</div>
            <div class="metric-value gold">Rp {int(latest_row['Retail_Price']):,} / gr</div>
            <div class="metric-delta {delta_class}">
                {sign}Rp {int(retail_change):,} ({sign}{retail_pct:.2f}%) today
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    delta_class = "delta-up" if spot_change >= 0 else "delta-down"
    sign = "+" if spot_change >= 0 else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">📈 Global Spot Price (IDR/gr equivalent)</div>
            <div class="metric-value blue">Rp {int(latest_row['Spot_IDR_Gram']):,} / gr</div>
            <div class="metric-delta {delta_class}">
                {sign}Rp {int(spot_change):,} ({sign}{spot_pct:.2f}%) today
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">⚖️ Physical Spread (Margin)</div>
            <div class="metric-value orange">{active_spread_pct:.2f}%</div>
            <div class="metric-delta">
                Premium: +Rp {int(spread_diff):,} above spot
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- ENGINE FORECAST ORCHESTRATION -----------------
with st.spinner("Re-training SARIMAX model and generating predictions..."):
    # 1. Fit Spot Model
    forecaster = GoldForecaster(order=(p_order, d_order, q_order))
    spot_series = db_df.set_index('Date')['Spot_IDR_Gram']
    forecaster.fit(spot_series)
    
    # 2. Get Spot Forecasts (7 steps)
    spot_forecast_df = forecaster.forecast(steps=7)
    
    # 3. Apply Margin Correction (Spread engine)
    retail_forecast_df = apply_spread_correction(
        spot_forecast_df, 
        spreads, 
        use_latest=use_latest_spread
    )

# ----------------- VISUALIZATION ENGINE (MATPLOTLIB) -----------------
st.markdown("### 📊 Historical Spot, Retail & 7-Day Forecasting")

# Prep data for plotting
plot_df = db_df.tail(lookback_days).copy()
plot_df.set_index('Date', inplace=True)

# Generate Matplotlib Figure
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 6.5), facecolor='#0E1117')
ax.set_facecolor('#111827')

# Color codes
COLOR_RETAIL = '#C9962C' # Retail Price Gold
COLOR_SPOT = '#185FA5'   # Spot Blue
COLOR_SPREAD = '#D85A30' # Spread Orange/Coral
COLOR_GRID = '#1F2937'

# Plot Historical Data
ax.plot(plot_df.index, plot_df['Retail_Price'], color=COLOR_RETAIL, label='Historical Retail (Beli)', linewidth=2.5)
ax.plot(plot_df.index, plot_df['Spot_IDR_Gram'], color=COLOR_SPOT, label='Historical Spot IDR/gr', linewidth=1.5, alpha=0.85)

# Plot Forecast Data
forecast_index = retail_forecast_df.index
ax.plot(forecast_index, retail_forecast_df['Retail_Forecast'], color=COLOR_RETAIL, linestyle='--', marker='o', markersize=6, label='Forecasted Retail (Beli)', linewidth=2)
ax.plot(forecast_index, retail_forecast_df['Forecast'], color=COLOR_SPOT, linestyle='--', marker='x', markersize=6, label='Forecasted Spot IDR/gr', linewidth=1.5, alpha=0.8)

# Shading for confidence intervals
ax.fill_between(
    forecast_index,
    retail_forecast_df['Retail_Lower_Bound'],
    retail_forecast_df['Retail_Upper_Bound'],
    color=COLOR_RETAIL,
    alpha=0.15,
    label='Retail 95% Confidence Band'
)

# Vertical line separating history and forecast
ax.axvline(x=plot_df.index[-1], color='#4B5563', linestyle=':', label='Forecast Horizon Boundary', alpha=0.8)

# Styling details
ax.set_title("HRTA Gold Prices: Historical Spot vs. Retail & Forecast Horizon", fontsize=14, color='#F3F4F6', pad=15, fontweight='semibold')
ax.set_ylabel("Price (IDR / Gram)", fontsize=11, color='#9CA3AF', labelpad=10)
ax.set_xlabel("Date", fontsize=11, color='#9CA3AF', labelpad=10)

ax.grid(True, which='both', color=COLOR_GRID, linestyle='-', linewidth=0.5)

# Format dates nicely on the X axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=25, color='#9CA3AF')
plt.yticks(color='#9CA3AF')

# Formatting values to rupiah string format on Y-axis
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"Rp {int(x):,}"))

# Legend alignment
ax.legend(facecolor='#1E293B', edgecolor='#C9962C', labelcolor='#F3F4F6', loc='upper left', framealpha=0.9, fontsize=10)

# Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#374151')
ax.spines['bottom'].set_color('#374151')

plt.tight_layout()

# Render Matplotlib figure in streamlit
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- DETAILED FORECAST TABLE -----------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 📋 7-Day Retail Pricing Forecast Table")
    
    # Format forecast table values for elegant display
    display_df = retail_forecast_df.copy()
    display_df.index = display_df.index.strftime('%Y-%m-%d (%A)')
    display_df.index.name = "Forecast Date"
    
    # Format as currency
    formatted_df = pd.DataFrame(index=display_df.index)
    formatted_df['Forecast Spot IDR/gr'] = display_df['Forecast'].apply(lambda x: f"Rp {int(x):,}")
    formatted_df['Forecast Retail Beli/gr'] = display_df['Retail_Forecast'].apply(lambda x: f"Rp {int(x):,}")
    formatted_df['Retail Lower Bound (95%)'] = display_df['Retail_Lower_Bound'].apply(lambda x: f"Rp {int(x):,}")
    formatted_df['Retail Upper Bound (95%)'] = display_df['Retail_Upper_Bound'].apply(lambda x: f"Rp {int(x):,}")
    formatted_df['Forecast Buyback/gr'] = display_df['Buyback_Forecast'].apply(lambda x: f"Rp {int(x):,}")
    
    st.dataframe(formatted_df, use_container_width=True)
    
    # Download forecast csv option
    csv_data = retail_forecast_df.to_csv()
    st.download_button(
        label="📥 Download Forecast Data (CSV)",
        data=csv_data,
        file_name=f"hrta_forecast_{datetime.date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with right_col:
    st.markdown("### 🔍 Model Diagnostic Info")
    aic_val = f"{forecaster.model_res.aic:.2f}" if forecaster.model_res else "N/A"
    bic_val = f"{forecaster.model_res.bic:.2f}" if forecaster.model_res else "N/A"
    st.markdown(
        f"""
        <div style='background-color: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.9rem;'>
            <p><b>Model Selected:</b> SARIMAX({p_order}, {d_order}, {q_order})</p>
            <p><b>Akaike Info Criterion (AIC):</b> {aic_val}</p>
            <p><b>Bayesian Info Criterion (BIC):</b> {bic_val}</p>
            <p><b>Recent Gold Spot (GC=F):</b> USD {latest_row['Gold_USD_Oz']:.2f} / oz</p>
            <p><b>Recent Exchange Rate (IDR=X):</b> Rp {latest_row['USD_IDR']:.2f} / USD</p>
            <hr style='border-color: rgba(255, 255, 255, 0.05);'>
            <p style='color: #94a3b8; font-size: 0.85rem; line-height: 1.4;'>
                The <b>Spot Forecaster Engine</b> fits statsmodels SARIMAX on historical spot IDR prices.
                The <b>Spread Corrector Engine</b> applies a physical premium correction (currently <b>{active_spread_pct:.2f}%</b>) 
                to determine physical retail buyback and buy prices.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------- FOOTER -----------------
st.markdown(
    """
    <div class="footer">
        HRTA Gold Price Forecaster Application • Built using python/statsmodels SARIMAX & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
