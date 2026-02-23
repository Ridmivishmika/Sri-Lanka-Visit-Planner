# 🌴 Sri Lanka Visit Planner — ML Assignment

A machine learning project that predicts monthly tourist arrivals to Sri Lanka and provides personalised visit recommendations through an interactive Streamlit web app.

---

## 📁 Project Structure

```
ML Assigmnt/
│
├── generate_data.py      # Sri Lanka tourism dataset 
├── preprocess.py         # Data cleaning, typo fixing, encoding & imputation
├── train_model.py        # LightGBM model training, tuning, evaluation & SHAP
├── app.py                # Streamlit web application 
│
├── data/
│   ├── sl_tourism_raw.csv        # Dataset (output of generate_data.py)
│   ├── sl_tourism_clean.csv      # Cleaned & encoded dataset (output of preprocess.py)
│   ├── label_encoders.pkl        # Saved LabelEncoders for categorical columns
│   └── lgbm_model.pkl            # Trained LightGBM model + metrics
│
└── outputs/
    ├── 1_target_distribution.png
    ├── 2_seasonal_pattern.png
    ├── 3_arrivals_by_year.png
    ├── 4_correlation_heatmap.png
    ├── 5_actual_vs_predicted.png
    ├── 6_feature_importance.png
    ├── 7_shap_summary.png
    ├── 8_shap_bar.png
    └── 9_shap_waterfall.png
```

---

## 🔄 Pipeline Overview

```
generate_data.py  →  preprocess.py  →  train_model.py  →  app.py
  (raw dataset)       (clean data)      (trained model)    (web app)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib seaborn streamlit
```

---

## 🚀 How to Run

Run the scripts **in order**:

### Step 1 — Generate the dataset
```bash
python generate_data.py
```
Creates `data/sl_tourism_raw.csv` (1 500+ synthetic tourist records, 2015–2023).

### Step 2 — Preprocess the data
```bash
python preprocess.py
```
- Fixes country-name typos
- Removes duplicate rows
- Imputes missing values (median strategy)
- Label-encodes categorical columns
- Saves `data/sl_tourism_clean.csv` and `data/label_encoders.pkl`

### Step 3 — Train the model
```bash
python train_model.py
```
- Trains a **LightGBM Regressor** with early stopping
- Runs **GridSearchCV** hyperparameter tuning
- Evaluates on held-out test set (RMSE, MAE, R²)
- Generates **9 EDA + evaluation + SHAP plots** in `outputs/`
- Saves `data/lgbm_model.pkl`

### Step 4 — Launch the web app
```bash
streamlit run app.py
```
Opens the interactive **Sri Lanka Visit Planner** in your browser.

---

## 🌐 Web App Features

| Feature | Description |
|---|---|
| **Arrival Prediction** | Predicts expected monthly tourist arrivals for a chosen year, month, country, purpose, place & accommodation |
| **Visit Verdict** | Rates the visit as *Excellent / Good / Decent / Not Ideal* based on predicted arrivals |
| **Feature Analysis** | Breaks down each selection (month, country, place, accommodation, season, safety) with ✅ / ⚠️ / ❌ ratings |
| **Best Combination** | Recommends the ideal month, place & season for the chosen travel purpose |
| **Trend Chart** | Shows when tourists from the selected country historically visit Sri Lanka most |

---

## 🤖 Model Details

| Item | Value |
|---|---|
| **Algorithm** | LightGBM Regressor |
| **Target** | `Monthly_Arrivals` |
| **Features** | Year, Month, Country of Origin, Purpose, Primary District, Accommodation Type, Duration Days, Group Size, Age, Prior Visits SL, Season, Crisis Period |
| **Train/Val/Test Split** | 70 / 15 / 15 |
| **Tuning** | GridSearchCV (3-fold CV) over `num_leaves`, `learning_rate`, `n_estimators` |
| **Explainability** | SHAP TreeExplainer (summary, bar & waterfall plots) |

---

## 📊 Dataset

The dataset is **synthetically generated** to simulate realistic Sri Lanka tourism patterns (2015–2023):

- **1 500 records** sampled with realistic country shares, seasonal weights, and crisis multipliers
- **Crisis periods** modelled: Easter Attacks (Apr–Jun 2019), COVID-19 (2020–2021)
- **Intentional noise** introduced: typos, missing values, duplicate rows (for preprocessing practice)

### Key columns

| Column | Type | Description |
|---|---|---|
| `Year` | int | Visit year (2015–2023) |
| `Month` | int | Visit month (1–12) |
| `Country_of_Origin` | str | Tourist's home country |
| `Purpose` | str | Travel purpose (Beach, Cultural, Adventure, etc.) |
| `Primary_District` | str | Main destination district |
| `Accommodation_Type` | str | Budget Hotel, 3-Star, 4-Star, etc. |
| `Duration_Days` | float | Length of stay |
| `Group_Size` | int | Number of travellers |
| `Age` | float | Traveller's age |
| `Prior_Visits_SL` | int | Number of previous visits |
| `Season` | str | Dry Season / Wet Season |
| `Crisis_Period` | int | 1 = historical crisis, 0 = normal |
| `Monthly_Arrivals` | int | **Target** — estimated monthly arrivals |

---

## 📈 Output Plots

| File | Description |
|---|---|
| `1_target_distribution.png` | Histogram of monthly arrivals |
| `2_seasonal_pattern.png` | Average arrivals by month (seasonality) |
| `3_arrivals_by_year.png` | Year-on-year trend (showing crisis drops) |
| `4_correlation_heatmap.png` | Feature correlation matrix |
| `5_actual_vs_predicted.png` | Actual vs Predicted scatter plot |
| `6_feature_importance.png` | LightGBM built-in feature importance |
| `7_shap_summary.png` | SHAP beeswarm summary plot |
| `8_shap_bar.png` | Mean \|SHAP\| bar chart (global importance) |
| `9_shap_waterfall.png` | SHAP waterfall for a single prediction |

---


