# EV Price Predictor - Machine Learning 🚗

An end-to-end Machine Learning web application designed to predict the fair market value of Electric Vehicles (EVs). Built with **Python, Flask, XGBoost**, and an embedded **SQLite** database.

## 💾 The Concept
Predicting the price of a used EV isn't as simple as drawing a straight line. Factors like battery degradation and autonomy level interact in non-linear ways. This application takes user input, engineers advanced features (like *Efficiency Score*), and feeds it into a pre-trained **XGBoost Regressor** to determine a data-driven price.

## 🏗️ Architecture & Tech Stack
This project operates as a full-stack ML application:
* **Machine Learning:** `scikit-learn`, `xgboost`, `pandas`
* **Web Framework:** `Flask` (Python)
* **Database:** `SQLite` (via `Flask-SQLAlchemy`)
* **Frontend:** HTML5, CSS3

## 🧠 How the Machine Learning Works
This project solves a **Supervised Regression** problem. 

1. **Algorithm:** We evolved from Linear Regression (which underfitted the data) to **eXtreme Gradient Boosting (XGBoost)**. XGBoost builds hundreds of sequential decision trees, with each new tree correcting the errors of the previous one.
2. **Feature Engineering:** We derived two hidden signals before feeding the data to the model.
3. **Scaling:** A `StandardScaler` normalizes massive values (like Range: 500) and tiny values (like Safety: 5) to prevent algorithmic bias.

## 🚀 Getting Started (Run Locally)

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ev-price-predictor.git](https://github.com/your-username/ev-price-predictor.git)
cd ev-price-predictor
