# 📊 Sales Forecasting & Demand Prediction

This project builds a machine learning pipeline to predict future **product demand** using historical **sales data** and external variables. It is designed to help businesses optimize inventory, reduce stockouts, and improve revenue forecasting.

---

## 📁 Dataset

- Historical sales data of multiple products across different stores.
- Features include: product IDs, date, units sold, promotions, and external factors (e.g., seasonality).
- Target: **Units Sold (regression problem)**

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- XGBoost
- Scikit-learn
- Matplotlib, Seaborn
- Flask (for optional deployment)

---

## 🔍 Project Pipeline

1. **EDA** – Trend & seasonality detection, null handling, and data cleaning
2. **Feature Engineering** – Lag features, rolling stats, and datetime transformations
3. **Data Preprocessing** – Scaling, encoding categorical variables
4. **Modeling** – XGBoost Regressor with hyperparameter tuning
5. **Evaluation** – R² score, RMSE, MAE, and visualizations
6. **Deployment** – Simple web API using Flask (optional module)

---

## 🔢 Model Performance

| Metric | Score |
|--------|-------|
| R²     | **0.87** |
| RMSE   | 73.3 |
| MAE    | 69.5 |

> Achieved strong generalization using cross-validation.

---

## 🖼️ Visuals

- Feature Importance Plot
- Predicted vs Actual Sales
- Residual Distribution
- Demand Forecasting Curves

---

## 🧠 Skills Demonstrated

- Time series & tabular forecasting
- Regression model building & tuning
- Feature engineering for demand patterns
- Real-world problem framing
- Python ML deployment (optional)

---

## 📌 Disclaimer

> Dataset used is synthetic or anonymized for privacy reasons.

---

## 🔗 Author

**Youssef Ali Manaa**  
[LinkedIn](https://www.linkedin.com/in/youssef-ali-manaa) • [GitHub](https://github.com/youssef2003ali)
