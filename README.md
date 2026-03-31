# Apple-Stock-Prediction


## 📱 App Demo

https://apple-stock-prediction-dotx9tv4d5trz4vkeccjwj.streamlit.app/

![App Demo](./appdemo.pdf)

## 📄 Research Paper

You can read the full research paper for this project here:  

[Apple Stock Research Paper (PDF)](./Apple-stock-ResearchPaper.pdf)



This project is an end-to-end Machine Learning application that predicts the next day’s closing price of Apple (AAPL) using historical stock data.

It combines data analysis, feature engineering, and predictive modeling with an interactive Streamlit dashboard.

🎯 Project Objective

The goal of this project is to:

Analyze historical stock data
Build predictive models for short-term forecasting
Compare multiple models
Deploy an interactive web application for real-time predictions

🧠 Models Used
Random Forest
Gradient Boosting
Ridge Regression
USED FOR APP
Linear Regression
Bayesian Ridge Regression
Both models were selected due to their strong performance on structured financial data.

📊 Key Results
Model	MAE	R² Score
Linear Regression	~1.65	~0.957
Bayesian Ridge	~1.67	~0.956

📌 Insight:

Both models perform extremely well
Predictions closely match actual stock prices
Linear-based models outperform complex models in this setup

⚙️ Features Used
Open, High, Low, Volume
Previous Day Prices (Prev_Close, Prev_Open)
Moving Averages (MA_5, MA_10)
Volatility
Daily Return

📌 Observation:

Daily Return is the most influential feature
Other features have minimal impact

📈 Project Structure
apple-stock-prediction/



├── app.py        
# Streamlit app (UI + visualization)
├── model.py       
# ML training & prediction logic
├── apple_stock.ipynb 
# EDA + model development notebook
├── requirements.txt  
# Dependencies
└── README.md       
# Project documentation

📂 File Explanation

📊 apple_stock.ipynb


This is the research and development notebook.

It contains:

Data loading using yfinance
Exploratory Data Analysis (EDA)
Feature engineering
Model training (Linear, Bayesian, Random Forest, etc.)
Performance comparison

👉 This file shows how the model was built and tested

⚙️ model.py

This file contains the core machine learning logic.

It handles:

Data preprocessing
Feature engineering
Train-test split
Model training (Linear + Bayesian)
Prediction functions

👉 This is the backend brain of the app

🌐 app.py

This is the Streamlit web application.

It provides:

Model performance metrics
Actual vs predicted visualization
Residual analysis


📦 Requirements
streamlit
pandas
numpy
matplotlib
scikit-learn
yfinance

💡 Key Insight 

Even though advanced models like Random Forest and Gradient Boosting were tested:

Simple linear models performed better
Why?
Stock prices in short-term follow strong linear trends
Features are highly correlated
Complex models overcomplicate simple patterns

🔮 Future Improvements

Add multiple stock selection (TSLA, MSFT, etc.)

Add real-time stock API

Improve UI with interactive charts

🏁 Conclusion

This project demonstrates that:

Strong feature engineering > complex models

Simpler models can outperform advanced ones

End-to-end ML pipelines are essential for real-world applications

