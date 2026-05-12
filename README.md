# Industrial Filter Predictive Maintenance

## Project Overview

This project explores predictive maintenance modeling for industrial air filtration systems using machine learning and physics-informed feature engineering.

The dataset contains lifecycle sensor data collected from industrial air filters operating under different dust and flow conditions. The primary objective was to investigate whether Remaining Useful Life (RUL) and degradation behavior could be predicted from operational sensor measurements.

During the project, a major modeling challenge emerged:

The training dataset contained right-censored maintenance trajectories, meaning many filters were replaced before actual failure occurred. This introduced strong temporal bias when pseudo-RUL labels were engineered directly from operational time.

As a result, the project evolved beyond standard modeling into an investigation of:

* target engineering
* temporal bias
* target leakage
* degradation dynamics
* physics-informed predictive maintenance modeling

---

# Dataset Information

Dataset:
Creation of Publicly Available Data Sets for Prognostics and Diagnostics Addressing Data Scenarios Relevant to Industrial Applications

Authors:

* Hagmeyer, S.
* Mauthe, F.
* Zeiler, P.

Failure Condition:
A filter is considered failed when:

Differential Pressure >= 600 Pa

The dataset contains:

* Right-censored training trajectories
* Run-to-failure test trajectories
* Sensor measurements across multiple operating conditions

---

# Project Structure

```text
Industrial Filter Predictive Maintenance/
│
├── Data/
├── models/
├── Notebooks/
│   ├── 01_data_understanding_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_physics_informed_target_engineering.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Workflow

## 1. Exploratory Data Analysis

The first notebook explores:

* lifecycle degradation behavior
* pressure progression trends
* flow rate distributions
* dust characteristics
* operational variability across filters

Key observations:

* Filter degradation is highly nonlinear
* Differential pressure accelerates rapidly near failure
* Dust characteristics strongly influence degradation rate
* Fine dust causes faster clogging than coarse dust

---

## 2. Feature Engineering

Several degradation-aware features were engineered, including:

* pressure growth rate
* pressure acceleration
* rolling pressure statistics
* cumulative degradation measures
* normalized lifecycle progression
* operational grouping variables

The objective was to capture degradation dynamics rather than relying purely on raw sensor values.

---

## 3. Initial RUL Modeling

Initial Random Forest and XGBoost models were trained using pseudo-RUL labels generated from:

RUL = max_time - Time

The models achieved very high internal validation performance.

However, evaluation on the official run-to-failure test dataset revealed a significant drop in generalization performance.

This investigation revealed that:

* lifecycle progression variables dominated prediction behavior
* the model relied heavily on temporal structure
* physically meaningful degradation features contributed less than expected

This indicated strong target bias caused by time-derived pseudo-RUL labels.

---

## 4. Physics-Informed Target Engineering

A second modeling investigation explored a physics-informed degradation target:

Target Capacity = 600 - Differential Pressure

This target formulation aligned prediction behavior more closely with physical degradation progression.

The notebook investigated:

* target engineering
* target leakage
* feature-target alignment
* degradation-dynamics prediction

After removing leakage-prone variables, the refined leakage-aware model demonstrated:

* strong predictive performance
* meaningful degradation-dynamic learning
* dominant importance of pressure growth behavior

Final refined model performance:

* R² Score: ~0.94
* RMSE: ~30

Most important predictive variables:

* pressure_growth_rate
* lifecycle progression
* normalized operational progression

---

# Key Findings

## 1. Target Engineering Strongly Influences Model Behavior

Time-derived pseudo-RUL labels caused models to prioritize lifecycle progression over physical degradation behavior.

---

## 2. Degradation Dynamics Contain Strong Predictive Information

When target formulation was aligned with physical degradation behavior, degradation-dynamic features became highly predictive.

---

## 3. Leakage-Aware Design Is Critical

Several engineered pressure features indirectly reconstructed the target itself.

Feature redesign was required to separate:

* physically meaningful predictive behavior
  from
* target leakage

---

# Models Used

* Random Forest Regressor
* XGBoost Regressor

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Joblib
* Jupyter Notebook

---

# Saved Artifacts

The final leakage-aware physics-informed model was exported using Joblib.

Saved files:

* physics_informed_rf_model.pkl
* physics_informed_features.pkl

---

# Future Improvements

Potential future extensions include:

* LSTM-based sequence modeling
* survival analysis
* physics-informed neural networks
* health index modeling
* anomaly detection approaches
* uncensored degradation modeling

---

# Repository

GitHub Repository:

[https://github.com/thatboypage/industrial-filter-predictive-maintenance](https://github.com/thatboypage/industrial-filter-predictive-maintenance)

---

# Author

Richard

Machine Learning | Predictive Maintenance | AI Engineering
