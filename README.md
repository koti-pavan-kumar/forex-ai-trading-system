# 🤖 AI-Powered Forex Trading System

A production-ready algorithmic trading system
built in Python that uses Machine Learning
to generate real-time BUY/SELL signals
for Forex currency pairs.

---

## 🎯 What It Does

- Connects to **MetaTrader 5** for live broker data
- Calculates **65 technical indicators** in real-time
- Predicts market direction using **XGBoost AI**
- Generates BUY/SELL signals with confidence scores
- Displays everything on a **professional dashboard**
- **Automatically records** paper trades
- **Walk-forward validation** (no data leakage)

---

## 📊 Model Performance

| Pair   | Accuracy | Timeframe | Status  |
|--------|----------|-----------|---------|
| GBPUSD | 55.2%    | H1        | ✅ Active |
| USDJPY | 50.4%    | H1        | ⚠️ Monitor |
| EURUSD | 38.0%    | H1        | ❌ Skip  |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| XGBoost | AI signal classifier |
| MetaTrader 5 | Live broker data |
| Streamlit | Interactive dashboard |
| Pandas / NumPy | Data processing |
| Plotly | Professional charts |
| Scikit-learn | Walk-forward validation |

---

## 📁 Project Structure
```
forex-ai/
│
├── data_source.py       # MT5 + yfinance connection
├── indicators.py        # 65 technical indicators
├── train_model.py       # AI model training (XGBoost)
├── live_signals.py      # Real-time signal generation
├── backtest.py          # Strategy backtesting engine
├── dashboard.py         # Streamlit visual dashboard
├── paper_trading.py     # Paper trade journal
├── main.py              # Master automation script
├── morning_check.py     # Daily routine script
├── requirements.txt     # Python dependencies
└── .env.example         # Credentials template
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOURUSERNAME/forex-ai-trading-system.git
cd forex-ai-trading-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up credentials
```bash
cp .env.example .env
# Edit .env with your MT5 credentials
```

### 5. Run daily check
```bash
python morning_check.py
```

### 6. Open live dashboard
```bash
streamlit run dashboard.py
```

---

## 📈 System Architecture
```
MT5 Live Data
      ↓
65 Indicators (RSI, MACD, BB, EMA, ATR...)
      ↓
XGBoost AI Model
      ↓
BUY / SELL Signal + Confidence Score
      ↓
Entry Price + Stop Loss + Take Profit
      ↓
Streamlit Dashboard + Paper Trade Journal
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in:
```
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=ICMarketsSC-Demo
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
- Paper trade for minimum 3 months before real money
- Past backtesting performance does not guarantee future results
- Forex trading carries substantial risk of loss
- Never risk money you cannot afford to lose

---

## 👤 Author

**Pavan Kumar Koti**
AI Student — Pace Institute of Technology and Sciences
Internship Project — Learnmind.ai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yourusername)
```

---

### STEP 7 — Create .env.example File

Create a new file called `.env.example` in your `forex-ai` folder:
```
# Copy this file to .env and fill in your details
# Never share your actual .env file

MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=ICMarketsSC-Demo
ALPHA_VANTAGE_KEY=your_key_here
