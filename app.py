import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="Tesla Stock Price Prediction Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark theme and custom CSS styles
st.markdown("""
    <style>
    /* Gradient Title */
    .title-gradient {
        background: linear-gradient(45deg, #FF2E2E, #FF8008);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #B0B0B0;
        margin-bottom: 2rem;
    }
    
    /* Custom Card Design */
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #FF3B30;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    .metric-card-title {
        color: #A0A0A0;
        font-size: 0.9rem;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-card-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #707070;
        text-align: center;
        padding: 10px 0;
        font-size: 0.8rem;
        border-top: 1px solid #202020;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    # Load raw data for historical visualization
    raw_df = pd.read_csv("TSLA.csv")
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values('Date')
    
    # Load model predictions
    preds_df = pd.read_csv("models/predictions.csv")
    preds_df['Date'] = pd.to_datetime(preds_df['Date'])
    
    # Load metrics
    metrics_df = pd.read_csv("models/evaluation_metrics.csv")
    
    # Load training history
    lstm_hist = pd.read_csv("models/lstm_history.csv")
    rnn_hist = pd.read_csv("models/rnn_history.csv")
    
    # Load grid search
    grid_df = pd.read_csv("models/grid_search_results.csv")
    
    return raw_df, preds_df, metrics_df, lstm_hist, rnn_hist, grid_df

# Check if model outputs exist
if not os.path.exists("models/predictions.csv"):
    st.error("Model files and predictions not found. Please run the training pipeline script first!")
    st.stop()

# Load data
raw_df, preds_df, metrics_df, lstm_hist, rnn_hist, grid_df = load_data()

# App Header
st.markdown("<div class='title-gradient'>Tesla Stock Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>A Premium Deep Learning Dashboard evaluating SimpleRNN and LSTM Multi-Step Forecasts</div>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=100)
st.sidebar.markdown("### Model Configuration")

# Selection boxes
model_type = st.sidebar.selectbox("Select Model Architecture", ["LSTM", "SimpleRNN"])
horizon_type = st.sidebar.selectbox("Select Forecast Horizon", ["1-Day Ahead", "5-Day Ahead", "10-Day Ahead"])

# Get key strings for columns
horizon_key = "1d" if "1" in horizon_type else ("5d" if "5" in horizon_type else "10d")
model_key = "LSTM" if model_type == "LSTM" else "RNN"
col_name = f"{model_key}_{horizon_key}"

st.sidebar.markdown("---")
st.sidebar.markdown("### About the Project")
st.sidebar.info(
    "This dashboard evaluates multi-step deep learning models trained on 2416 days of historical Tesla stock prices. "
    "Models look back at a 60-day window to predict the future 10-day price path."
)

# Create layout Tabs
tab_analytics, tab_forecast, tab_comparison, tab_report = st.tabs([
    "📈 Historical Analytics", 
    "⚡ Deep Learning Forecast", 
    "📊 Model Comparison & Tuning",
    "📝 Executive Summary"
])

# Tab 1: Historical Stock Analytics
with tab_analytics:
    st.markdown("### Historical Stock Analytics & Volatility")
    
    # Add Moving Averages
    raw_df['SMA_50'] = raw_df['Adj Close'].rolling(window=50).mean()
    raw_df['SMA_200'] = raw_df['Adj Close'].rolling(window=200).mean()
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    # Extract latest details
    latest_date = raw_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    latest_close = f"${raw_df['Adj Close'].iloc[-1]:.2f}"
    latest_volume = f"{raw_df['Volume'].iloc[-1]:,}"
    daily_change = (raw_df['Adj Close'].iloc[-1] - raw_df['Adj Close'].iloc[-2]) / raw_df['Adj Close'].iloc[-2] * 100
    change_color = "green" if daily_change >= 0 else "red"
    
    col_metric1.markdown(f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Latest Trading Date</div>
            <div class='metric-card-value'>{latest_date}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_metric2.markdown(f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Latest Adjusted Close</div>
            <div class='metric-card-value'>{latest_close}</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_metric3.markdown(f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Daily Change (%)</div>
            <div class='metric-card-value' style='color:{change_color};'>{daily_change:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    col_metric4.markdown(f"""
        <div class='metric-card'>
            <div class='metric-card-title'>Trading Volume</div>
            <div class='metric-card-value'>{latest_volume}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Interactive Plotly Close Price & Moving Averages
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=raw_df['Date'], y=raw_df['Adj Close'], name='Adjusted Close', line=dict(color='royalblue', width=1.5)))
    fig_price.add_trace(go.Scatter(x=raw_df['Date'], y=raw_df['SMA_50'], name='50-Day SMA', line=dict(color='orange', width=2, dash='dash')))
    fig_price.add_trace(go.Scatter(x=raw_df['Date'], y=raw_df['SMA_200'], name='200-Day SMA', line=dict(color='crimson', width=2, dash='dot')))
    
    fig_price.update_layout(
        title="Tesla Adjusted Closing Prices with Technical Indicators (SMA)",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volume and Returns Distributions
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        fig_vol = px.bar(raw_df, x='Date', y='Volume', title="Historical Trading Volume", color_discrete_sequence=['darkorange'])
        fig_vol.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col_graph2:
        raw_df['Daily Returns'] = raw_df['Adj Close'].pct_change() * 100
        fig_dist = px.histogram(raw_df, x='Daily Returns', nbins=100, title="Daily Percentage Returns Distribution", color_discrete_sequence=['teal'])
        fig_dist.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

# Tab 2: Deep Learning Forecast
with tab_forecast:
    st.markdown(f"### {model_type} Model Prediction — {horizon_type}")
    
    # Filter out metrics for display
    model_label = "LSTM" if model_type == "LSTM" else "RNN"
    selected_metric = metrics_df[metrics_df['Horizon'] == f"{model_label}_{horizon_type}"].iloc[0]
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Mean Squared Error (MSE)", f"{selected_metric['MSE']:.2f}")
    col_m2.metric("Root Mean Squared Error (RMSE)", f"{selected_metric['RMSE']:.2f}")
    col_m3.metric("Mean Absolute Error (MAE)", f"{selected_metric['MAE']:.2f}")
    col_m4.metric("R² Score (Coefficient of Determination)", f"{selected_metric['R2']:.4f}")
    
    # Interactive Plotly actual vs predicted
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=preds_df['Date'], y=preds_df['Actual_Close'], name='Actual Price', line=dict(color='white', width=2)))
    fig_forecast.add_trace(go.Scatter(x=preds_df['Date'], y=preds_df[col_name], name=f'Predicted ({model_type})', line=dict(color='#FF3B30', width=2)))
    
    fig_forecast.update_layout(
        title=f"Actual vs. Predicted Tesla Closing Prices ({model_type} - {horizon_type})",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Show prediction data table
    st.markdown("#### Forecast Data Table")
    display_df = preds_df[['Date', 'Actual_Close', col_name]].copy()
    display_df.columns = ['Date', 'Actual Closing Price ($)', 'Model Predicted Price ($)']
    display_df['Prediction Error ($)'] = display_df['Actual Closing Price ($)'] - display_df['Model Predicted Price ($)']
    st.dataframe(display_df.style.format({
        'Actual Closing Price ($)': '{:.2f}',
        'Model Predicted Price ($)': '{:.2f}',
        'Prediction Error ($)': '{:+.2f}'
    }), use_container_width=True)

# Tab 3: Model Comparison & Tuning
with tab_comparison:
    st.markdown("### SimpleRNN vs. LSTM Model Performance Comparison")
    
    # Show side-by-side metric tables
    st.markdown("#### Performance Metrics Breakdown")
    styled_metrics = metrics_df.style.format({
        'MSE': '{:.2f}',
        'RMSE': '{:.2f}',
        'MAE': '{:.2f}',
        'R2': '{:.4f}'
    })
    st.dataframe(styled_metrics, use_container_width=True)
    
    # Plotly Bar Chart comparing R2 score
    fig_r2 = px.bar(
        metrics_df, 
        x='Horizon', 
        y='R2', 
        color='Horizon',
        title="R² Score Comparison across Architectures and Horizons",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_r2.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # Loss curves and Hyperparameter tuning
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### Training Loss Curves")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=lstm_hist['epoch'], y=lstm_hist['lstm_loss'], name='LSTM Train Loss', line=dict(color='royalblue')))
        fig_loss.add_trace(go.Scatter(x=lstm_hist['epoch'], y=lstm_hist['lstm_val_loss'], name='LSTM Val Loss', line=dict(color='orange')))
        fig_loss.add_trace(go.Scatter(x=rnn_hist['epoch'], y=rnn_hist['rnn_loss'], name='RNN Train Loss', line=dict(color='crimson', dash='dash')))
        fig_loss.add_trace(go.Scatter(x=rnn_hist['epoch'], y=rnn_hist['rnn_val_loss'], name='RNN Val Loss', line=dict(color='gold', dash='dash')))
        
        fig_loss.update_layout(
            title="Epoch Loss Curve comparison",
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with col_c2:
        st.markdown("#### GridSearchCV LSTM Tuning Results")
        # Display the grid search table
        st.dataframe(grid_df.sort_values('val_loss').style.format({
            'learning_rate': '{:.4f}',
            'dropout_rate': '{:.2f}',
            'val_loss': '{:.6f}'
        }), use_container_width=True)
        
        best_row = grid_df.loc[grid_df['val_loss'].idxmin()]
        st.success(
            f"**GridSearchCV Winner:** LSTM Units: **{int(best_row['units'])}**, "
            f"Dropout Rate: **{best_row['dropout_rate']:.2f}**, "
            f"Learning Rate: **{best_row['learning_rate']:.3f}**"
        )

# Tab 4: Project Executive Summary
with tab_report:
    st.markdown("### Executive Project Summary")
    st.markdown("""
    #### 1. Domain Background & Problem Statement
    In quantitative finance, predicting stock prices is highly challenging due to non-stationarity, market noise, and microstructural shifts. This project addresses the **Tesla (TSLA)** stock price movements by developing and comparing two sequential Deep Learning architectures:
    * **Simple Recurrent Neural Network (SimpleRNN)**
    * **Long Short-Term Memory (LSTM)**
    
    Both models are structured for **multi-step direct forecasting**, look back at a sliding window of **60 historical days** of adjusted closing prices, and output a vector predicting the next **10 consecutive days**. We evaluate performance at **1-day**, **5-day**, and **10-day** ahead horizons.
    
    #### 2. Key Findings & Analytics
    * **1-Day Ahead**: The models achieve exceptional predictive power with $R^2$ scores between **0.94 and 0.95**. Interestingly, SimpleRNN exhibits excellent short-term tracking of price directions because daily fluctuations are heavily influenced by short-term momentum.
    * **5-Day Ahead**: Accuracy degrades slightly to $R^2$ scores of **0.81 - 0.85**. 
    * **10-Day Ahead**: Over a 2-week trading horizon, the models capture macro-level price curves rather than exact day-to-day noise, resulting in $R^2$ scores of **0.66 - 0.67**. At this horizon, LSTM is structurally more stable as it mitigates vanishing gradients via input, forget, and output gating mechanisms.
    
    #### 3. Real-World Business Applications
    * 🔹 **Automated & Algorithmic Trading**: Deploying the 1-day prediction model as a signal generator in systematic momentum or mean-reversion trading systems.
    * 🔹 **Risk Management**: Utilizing 5-day and 10-day price path predictions to adjust options hedge parameters (e.g. dynamic delta-hedging) and evaluate portfolio value-at-risk.
    * 🔹 **Macro Financial Analysis**: Correlating model prediction errors with sudden macroeconomic announcements (e.g. Fed interest rate hikes, inflation index releases, or supply chain shocks).
    """)

# Footer
st.markdown("<div class='footer'>Tesla Stock Prediction Project Dashboard | Created for Financial Analytics Deep Learning Module</div>", unsafe_allow_html=True)
