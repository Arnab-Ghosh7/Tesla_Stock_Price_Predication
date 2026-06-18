# ⚡ Tesla Stock Price Prediction Dashboard

A premium quantitative finance dashboard comparing **LSTM** and **SimpleRNN** neural network architectures for multi-step stock price forecasting. Built with Python, TensorFlow/Keras, and Streamlit.

---

## 📖 Project Overview & Background
Predicting stock prices is a complex challenge in quantitative finance due to non-stationarity, market noise, and microstructural shifts. This project trains and evaluates sequential Deep Learning models on historical **Tesla (TSLA)** stock prices.




---

## 🛠️ Tech Stack & Key Libraries
- **Deep Learning Framework:** TensorFlow / Keras
- **Web App Dashboard:** Streamlit
- **Data Analysis & Preprocessing:** Pandas, NumPy, Scikit-learn
- **Data Visualization:** Plotly (interactive charts), Matplotlib, Seaborn

---

## 📂 Project Directory Structure
```text
Tesla Stock Price Predication/
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── TSLA.csv                        # Historical Tesla stock dataset
├── app.py                          # Streamlit dashboard application
├── pipeline.py                     # Model training and tuning pipeline
├── requirements.txt                # Python package dependencies
└── models/                         # Trained model artifacts & results
    ├── lstm_model.h5               # Saved LSTM model weights
    ├── rnn_model.h5                # Saved SimpleRNN model weights
    ├── scaler.pkl                  # MinMaxScaler fitted on training data
    ├── evaluation_metrics.csv      # Out-of-sample metrics (MSE, RMSE, MAE, R²)
    ├── grid_search_results.csv     # GridSearchCV hyperparameter tuning history
    ├── lstm_history.csv            # Training history for LSTM
    ├── rnn_history.csv             # Training history for SimpleRNN
    └── predictions.csv             # Historical test set predictions
```

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
Ensure you have **Python 3.8 - 3.11** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/Arnab-Ghosh7/Tesla_Stock_Price_Predication.git
cd Tesla_Stock_Price_Predication
```

### 3. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📈 Running the Project

### Phase 1: Train the Models & Run the Pipeline
If you want to retrain the models and regenerate all evaluation reports:
```bash
python pipeline.py
```
*Note: This script will perform grid search hyperparameter tuning for LSTM, train both architectures, and output all evaluation files to the `models/` directory.*

### Phase 2: Start the Streamlit Dashboard
Launch the interactive visualization dashboard locally:
```bash
streamlit run app.py
```
Once executed, the dashboard will open automatically in your default browser (typically at `http://localhost:8501`).

---

## 📊 Dashboard Modules & Features
The interactive dashboard is organized into four distinct sections:
1. **📈 Historical Analytics:** Explores Tesla stock price history with 50-day and 200-day Simple Moving Averages, historical trading volumes, and daily percentage returns distribution.
2. **⚡ Deep Learning Forecast:** Compares actual TSLA closing prices with model-predicted lines for 1-day, 5-day, and 10-day horizons. Displays real-time error tracking and raw forecast tables.
3. **📊 Model Comparison & Tuning:** Highlights side-by-side performance indicators (MSE, RMSE, MAE, R²), visualizes epoch-by-epoch loss curves, and displays the hyperparameter tuning search grid.
4. **📝 Executive Summary:** Contextualizes the project's background, core findings, model degradation across forecast horizons, and real-world business use cases.

---

## 💡 Real-World Applications
- **Systematic & Algorithmic Trading:** Use short-term 1-day price path indicators as direction signals for momentum or mean-reversion trading.
- **Quantitative Risk Management:** Utilize 5-day and 10-day paths to estimate dynamic hedging parameters (e.g., options delta-hedging) and project portfolio Value-at-Risk (VaR).
- **Macro Sentiment Auditing:** Correlate specific days of large model prediction errors with real-world events, earnings reports, or Federal Reserve announcements.

---

*Project created for Financial Analytics & Sequential Deep Learning applications.*
