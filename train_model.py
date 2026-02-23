
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# ── Paths ───────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
CLEAN     = os.path.join(BASE, "data", "sl_tourism_clean.csv")
MODEL_OUT = os.path.join(BASE, "data", "lgbm_model.pkl")
OUT       = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

print("=" * 55)
print("  MODEL TRAINING — LightGBM Regressor")
print("=" * 55)

# ── Load clean data ─────────────────────────────────────────────────────────
df = pd.read_csv(CLEAN)
print(f"\nLoaded: {df.shape}")

FEATURES = ["Year","Month","Country_of_Origin","Purpose","Primary_District",
            "Accommodation_Type","Duration_Days","Group_Size","Age",
            "Prior_Visits_SL","Season","Crisis_Period"]
TARGET   = "Monthly_Arrivals"

X = df[FEATURES]
y = df[TARGET]

# ── Train / Val / Test split (70 / 15 / 15) ─────────────────────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_te, y_val, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)
print(f"Train: {X_tr.shape}  Val: {X_val.shape}  Test: {X_te.shape}")

# ── EDA Plots ────────────────────────────────────────────────────────────────
print("\n[EDA] Generating exploratory plots ...")

# Target distribution
plt.figure(figsize=(9,4))
plt.hist(y, bins=40, color="#4285F4", edgecolor="white", alpha=0.85)
plt.title("Distribution of Monthly Tourist Arrivals", fontsize=13, fontweight="bold")
plt.xlabel("Monthly Arrivals"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"1_target_distribution.png"),dpi=150); plt.close()

# Arrivals by month (seasonal)
monthly = df.groupby("Month")[TARGET].mean()
months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
plt.figure(figsize=(9,4))
plt.bar(months, monthly.values, color="#34A853", edgecolor="white", alpha=0.85)
plt.title("Average Monthly Arrivals — Seasonal Pattern", fontsize=13, fontweight="bold")
plt.xlabel("Month"); plt.ylabel("Avg Arrivals")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"2_seasonal_pattern.png"),dpi=150); plt.close()

# Arrivals by year
yearly = df.groupby("Year")[TARGET].mean()
plt.figure(figsize=(9,4))
plt.plot(yearly.index, yearly.values, marker="o", color="#EA4335", linewidth=2.5)
plt.fill_between(yearly.index, yearly.values, alpha=0.15, color="#EA4335")
plt.title("Average Monthly Arrivals by Year", fontsize=13, fontweight="bold")
plt.xlabel("Year"); plt.ylabel("Avg Arrivals"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"3_arrivals_by_year.png"),dpi=150); plt.close()

# Correlation heatmap
plt.figure(figsize=(10,8))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            annot_kws={"size":8})
plt.title("Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"4_correlation_heatmap.png"),dpi=150); plt.close()

print("   EDA plots saved (1-4).")

# ── LightGBM with early stopping ────────────────────────────────────────────
print("\n[MODEL] Training LightGBM with early stopping ...")
lgbm_es = lgb.LGBMRegressor(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    n_jobs=-1, verbose=-1,
)
lgbm_es.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)])
print(f"   Best iteration: {lgbm_es.best_iteration_}")

# ── Hyperparameter Tuning ────────────────────────────────────────────────────
print("\n[TUNING] GridSearchCV ...")
param_grid = {
    "num_leaves":   [31, 63],
    "learning_rate":[0.05, 0.10],
    "n_estimators": [200, 400],
}
grid = GridSearchCV(
    lgb.LGBMRegressor(subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1, verbose=-1),
    param_grid, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1
)
grid.fit(X_tr, y_tr)
best = grid.best_estimator_
print(f"   Best params: {grid.best_params_}")
print(f"   Best CV RMSE: {-grid.best_score_:,.0f}")

# ── Evaluation ───────────────────────────────────────────────────────────────
print("\n[EVAL] Test set metrics ...")
y_pred = best.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
mae  = mean_absolute_error(y_te, y_pred)
r2   = r2_score(y_te, y_pred)
print(f"   RMSE : {rmse:,.2f}")
print(f"   MAE  : {mae:,.2f}")
print(f"   R²   : {r2:.4f}")

# Actual vs Predicted
plt.figure(figsize=(7,6))
plt.scatter(y_te, y_pred, alpha=0.4, s=25, color="#4285F4")
mx = max(y_te.max(), y_pred.max())
plt.plot([0,mx],[0,mx],"r--",linewidth=2,label="Perfect fit")
plt.xlabel("Actual Arrivals"); plt.ylabel("Predicted Arrivals")
plt.title(f"Actual vs Predicted  (R²={r2:.3f})", fontsize=13, fontweight="bold")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,"5_actual_vs_predicted.png"),dpi=150); plt.close()

# Feature Importance
feat_imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values(ascending=True)
plt.figure(figsize=(9,5))
feat_imp.plot(kind="barh", color="#FBBC04", edgecolor="white")
plt.title("LightGBM Feature Importance", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"6_feature_importance.png"),dpi=150); plt.close()

print("   Evaluation plots saved (5-6).")

# ── SHAP ─────────────────────────────────────────────────────────────────────
print("\n[SHAP] Computing SHAP values ...")
n_shap    = min(500, len(X_te))
X_shap    = X_te.sample(n_shap, random_state=42)
explainer = shap.TreeExplainer(best)
shap_vals = explainer.shap_values(X_shap)

# Summary plot
plt.figure()
shap.summary_plot(shap_vals, X_shap, feature_names=FEATURES, show=False, plot_size=(10,6))
plt.title("SHAP Summary Plot", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"7_shap_summary.png"),dpi=150,bbox_inches="tight"); plt.close()

# Bar plot (mean |SHAP|)
plt.figure()
shap.summary_plot(shap_vals, X_shap, feature_names=FEATURES,
                  plot_type="bar", show=False, plot_size=(10,5))
plt.title("Mean |SHAP| — Global Feature Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"8_shap_bar.png"),dpi=150,bbox_inches="tight"); plt.close()

# Waterfall (single prediction)
shap_exp = explainer(X_shap.iloc[:1])
plt.figure()
shap.waterfall_plot(shap_exp[0], show=False)
plt.title("SHAP Waterfall — Single Prediction", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"9_shap_waterfall.png"),dpi=150,bbox_inches="tight"); plt.close()

print("   SHAP plots saved (7-9).")

# ── Save model ───────────────────────────────────────────────────────────────
artifacts = {"model": best, "features": FEATURES,
             "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2}}
with open(MODEL_OUT, "wb") as f:
    pickle.dump(artifacts, f)
print(f"\nModel saved → {MODEL_OUT}")

# ── Final summary ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  RESULTS SUMMARY")
print("="*55)
print(f"  RMSE    : {rmse:>12,.2f}  (avg error in arrivals)")
print(f"  MAE     : {mae:>12,.2f}  (absolute avg error)")
print(f"  R² Score: {r2:>12.4f}  (1.0 = perfect fit)")
print(f"\n  All plots  → outputs/")
print(f"  Model      → {MODEL_OUT}")
print("  Done ✓")
