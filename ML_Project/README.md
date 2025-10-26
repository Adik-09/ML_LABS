# 🧠 Stock Price Prediction using LSTM

This project builds a **Long Short-Term Memory (LSTM)** neural network to predict future stock prices using **historical financial data** fetched from Yahoo Finance.  
It also includes a simple chatbot-style interface for predicting stock trends interactively.

---

## 🚀 Features

- 📈 **Automatic Stock Data Download** via [Yahoo Finance (`yfinance`)](https://pypi.org/project/yfinance/)
- 🧮 **Feature Engineering** with:
  - Returns (`%` change)
  - Moving Averages (10 & 20 days)
  - Volatility (10-day rolling std)
  - RSI (Relative Strength Index)
- 🔢 **Data Normalization** using `MinMaxScaler`
- 🧠 **LSTM Neural Network** built with TensorFlow/Keras
- 💾 **Model & Scaler Persistence** (`.h5`, `.pkl`)
- 💬 **Chatbot Interface** for real-time ticker predictions

---

## 📂 Project Structure

```
stock_predict_model.ipynb
├── Data Preprocessing
│   ├── Download data using yfinance
│   ├── Compute features (Return, MA, Volatility, RSI)
│   ├── Scale features with MinMaxScaler
├── Model Training
│   ├── Prepare training windows (60-day lookback)
│   ├── Build 2-layer LSTM with Dropout
│   ├── Train model on historical data
├── Model Saving & Loading
│   ├── Save model + scalers
│   ├── load_stock_model(ticker)
├── Prediction
│   ├── predict_with_stock_model(ticker)
│   ├── Uses recent 6 months of stock data
└── CLI Chatbot
    └── Interactive text input for predictions
```

---

## 🧩 Dependencies

Install all required packages before running the notebook:

```bash
pip install yfinance numpy pandas scikit-learn tensorflow joblib
```

---

## ⚙️ How to Use

### 1. Train a Model
Run the notebook and set your desired stock ticker (e.g., `AAPL`, `TSLA`, etc.):

```python
ticker = "AAPL"
```

This will:
- Download 10 years of stock data
- Train an LSTM model
- Save model and scalers as:
  ```
  aapl_model.h5
  aapl_scaler.pkl
  aapl_target_scaler.pkl
  ```

---

### 2. Make Predictions
Once trained, you can predict using:

```python
predict_with_stock_model("AAPL")
```

The function:
- Loads your saved model
- Fetches recent data
- Predicts the next stock price (scaled back to actual range)

---

### 3. Chatbot Interface

At the end of the notebook:
```python
while True:
    user_input = input("You: ")
```
Type a stock ticker (e.g. `TSLA`) and get predictions interactively:
```
You: TSLA
Chatbot: Predicted closing price is $...
```
Type `exit` or `quit` to stop.

---

## 📊 Model Architecture

| Layer | Type  | Units | Activation | Notes |
|-------|--------|--------|-------------|--------|
| 1 | LSTM | 64 | — | Return sequences |
| 2 | Dropout | — | 0.2 | Regularization |
| 3 | LSTM | 64 | — | No return sequences |
| 4 | Dropout | — | 0.2 | Regularization |
| 5 | Dense | 1 | — | Final output neuron |

**Loss Function:** Mean Squared Error  
**Optimizer:** Adam  
**Epochs:** 30  
**Batch Size:** 32  

---

## 🧠 Future Improvements

- Add sentiment data or news analysis
- Train models for multiple tickers simultaneously
- Build a web dashboard (e.g., with Streamlit)
- Add evaluation metrics (MAE, RMSE, R²)
