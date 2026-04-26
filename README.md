# Industrial Filter Predictive Maintenance

## 📌 Overview
This project focuses on predicting the Remaining Useful Life (RUL) of industrial air filters used in gas-solid separation systems.

Traditional systems rely on **preventive maintenance**, where filters are replaced at fixed intervals regardless of their condition. This often leads to inefficiencies such as wasted filter life or unexpected failures.

This project demonstrates a transition to **predictive maintenance**, where machine learning models estimate RUL based on sensor data.

---

## 📊 Dataset
- Simulated industrial degradation process
- Failure threshold: 600 Pa differential pressure
- Right-censored data due to preventive maintenance
- 50 filter lifecycles

---

## 🔍 Key Insights
- Degradation is non-linear and accelerates near failure
- Dust characteristics strongly influence degradation
- Dataset is imbalanced toward early lifecycle stages

---

## 🔧 Feature Engineering
Key features include:
- Pressure rate (dP/dt)
- Normalized lifecycle time
- Rolling pressure mean
- Pressure ratio
- Interaction features

---

## 🤖 Modeling Results

| Model | MAE | RMSE | R² |
|------|-----|------|----|
| Random Forest | ~1.22 | ~4.51 | ~0.99 |
| XGBoost | ~1.70 | ~4.45 | ~0.99 |

Without lifecycle position:
- Performance drops significantly (MAE ~12, R² ~0.85)

---

## 🧠 Key Takeaway
> Model performance is driven more by feature engineering—especially lifecycle position—than by model complexity.

---

## 📊 Feature Importance
- Normalized time (~56%) dominates predictions
- Dust-related features are highly influential
- Pressure-based features contribute less than expected

---

## ⚠️ Limitations
- RUL is derived from observed lifecycle endpoints (right-censored data)
- Strong dependence on temporal features
- Limited representation of true failure behavior

---

## 🚀 Future Work
- Physics-informed modeling
- Survival analysis techniques
- Deep learning approaches for time-series

---

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, XGBoost

---

## 📬 Author
Richard Olanite