# 📈 Convolutional Transformer for Statistical Arbitrage

A replication project of a deep learning–based statistical arbitrage strategy using **CNN + Transformer** models,  
adapted to the **KOSPI equity market**.  
The goal is to extract complex mean-reversion and asymmetric trend structures from residual time series and evaluate their trading performance.

---

## 🚀 Overview

- **CNN**: Learns local patterns such as short-term spikes or reversals in residual time series  
- **Transformer**: Captures long-term dependencies by linking local patterns  
- **Advantage**: Detects nonlinear mean-reversion and asymmetric trends that OU / FFT filters miss  

**Trading Pipeline**
1. **Arbitrage Portfolio Construction** – Residuals via factor models (FF, PCA, IPCA)  
2. **Signal Extraction** – CNN + Transformer  
3. **Trading Allocation** – Feedforward NN to maximize Sharpe Ratio  

---

## 📊 Results (CRSP)

- **Sharpe Ratio > 4** in the best configuration  
- **CAGR ~20%** maintained  
- Strategy robust even during market stress periods (e.g., 2008 global crisis, COVID-19 shock)  

---

## 🛠 Methodology

- **Data**: CRSP daily returns, large-cap stocks (2000s–present)  
- **Factors**: Fama-French factor adaptation, PCA, IPCA  
- **Signals**: Residuals (30–60 day rolling window)  
- **Model**: CNN + Transformer → FFN allocation  
- **Execution**: Daily rebalancing, realistic transaction cost assumptions  

---

## 🔁 Replication

To replicate this project you will need:
- KOSPI stock price and fundamentals dataset  
- Factor models (FF, PCA, IPCA) implemented for Korean market  
- DL environment for CNN + Transformer (PyTorch / TensorFlow)  

---

## 📌 Usability

- Generalized statistical arbitrage model for the **Korean equity market**  
- Extendable to high-frequency intraday data, ETFs, derivatives  
- Can combine with SDF-based investing for market premium + alpha  
