import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle, os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
RAW    = os.path.join(BASE, "data", "sl_tourism_raw_v2.csv")
CLEAN  = os.path.join(BASE, "data", "sl_tourism_clean.csv")
ENC    = os.path.join(BASE, "data", "label_encoders.pkl")

print("=" * 55)
print("  PREPROCESSING — Sri Lanka Tourism Dataset")
print("=" * 55)

# ── 1. Load ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW)
print(f"\n[1] Raw data loaded")
print(f"    Rows: {len(df)}  |  Columns: {df.shape[1]}")
print(f"    Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ── 2. Fix Typos ─────────────────────────────────────────────────────────────
typo_map = {
    "Indai":         "India",
    "Germny":        "Germany",
    "United Kingdm": "United Kingdom",
    "Austraila":     "Australia",
    "Frnace":        "France",
}
before_unique = df["Country_of_Origin"].nunique()
df["Country_of_Origin"] = df["Country_of_Origin"].replace(typo_map)
after_unique  = df["Country_of_Origin"].nunique()
print(f"\n[2] Typos fixed in 'Country_of_Origin'")
print(f"    Unique countries before: {before_unique}  →  after: {after_unique}")

# ── 3. Remove Duplicates ─────────────────────────────────────────────────────
before_rows = len(df)
df.drop_duplicates(inplace=True)
after_rows  = len(df)
print(f"\n[3] Duplicates removed: {before_rows - after_rows} rows")
print(f"    Rows remaining: {after_rows}")

# ── 4. Impute Missing Values ──────────────────────────────────────────────────
dur_median = df["Duration_Days"].median()
age_median = df["Age"].median()
df["Duration_Days"].fillna(dur_median, inplace=True)
df["Age"].fillna(age_median, inplace=True)
print(f"\n[4] Missing values imputed (median strategy)")
print(f"    Duration_Days median: {dur_median:.1f} days")
print(f"    Age median:           {age_median:.1f} years")
print(f"    Missing after imputation: {df.isnull().sum().sum()}")

# ── 5. Encode Categoricals ───────────────────────────────────────────────────
CAT_COLS = ["Country_of_Origin", "Purpose", "Primary_District",
            "Accommodation_Type", "Season"]

le_dict = {}
print(f"\n[5] Label Encoding categorical columns:")
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"    {col:25s} → {len(le.classes_)} unique classes")

# Save encoders for use in Streamlit app
with open(ENC, "wb") as f:
    pickle.dump(le_dict, f)
print(f"\n    Encoders saved → {ENC}")

# ── 6. Save Cleaned Dataset ──────────────────────────────────────────────────
df.to_csv(CLEAN, index=False)
print(f"\n[6] Clean dataset saved → {CLEAN}")
print(f"    Final shape: {df.shape}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("PREPROCESSING COMPLETE")
print("=" * 55)
print(df.head(5).to_string())
print(f"\nData types:\n{df.dtypes}")
