import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Sri Lanka Visit Planner", page_icon="🌴", layout="centered")

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  body { font-family: 'Segoe UI', sans-serif; }
  .verdict-card { padding:1.4rem 1.6rem; border-radius:12px;
                  margin:1rem 0; border-left:6px solid; }
  .feat-row { display:flex; justify-content:space-between; align-items:center;
              padding:0.55rem 0.8rem; border-radius:8px; margin:0.3rem 0;
              font-size:0.97rem; }
  .feat-good  { background:#34A85318; border-left:4px solid #34A853; }
  .feat-warn  { background:#FBBC0418; border-left:4px solid #FBBC04; }
  .feat-bad   { background:#EA433518; border-left:4px solid #EA4335; }
  .tag-good   { background:#34A853; color:white; padding:2px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
  .tag-warn   { background:#FBBC04; color:#333; padding:2px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
  .tag-bad    { background:#EA4335; color:white; padding:2px 10px;
                border-radius:20px; font-size:0.8rem; font-weight:600; }
  .section-head { font-size:1.1rem; font-weight:700; margin:1.1rem 0 0.4rem; color:#222; }
  /* Blue Check button */
  button[kind="primaryFormSubmit"] {
      background-color: #1a73e8 !important;
      color: white !important;
      border: none !important;
  }
  /* Red Clear button */
  button[kind="secondaryFormSubmit"] {
      background-color: #EA4335 !important;
      color: white !important;
      border: none !important;
  }
  button[kind="secondaryFormSubmit"]:hover {
      background-color: #c5221f !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open(os.path.join(BASE, "data", "lgbm_model.pkl"), "rb") as f:
        art = pickle.load(f)
    with open(os.path.join(BASE, "data", "label_encoders.pkl"), "rb") as f:
        le = pickle.load(f)
    raw = pd.read_csv(os.path.join(BASE, "data", "sl_tourism_raw.csv"))
    return art, le, raw

artifacts, le_dict, raw_df = load_all()
model, FEATURES = artifacts["model"], artifacts["features"]

# ── Helpers ───────────────────────────────────────────────────────────────────
MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
FUTURE_YEARS   = list(range(2026, 2029))   # next 3 years only
CRISIS_PERIODS = {(2019,4),(2019,5),(2019,6)} | \
                 {(y,m) for y in [2020,2021] for m in range(1,13)}
PEAK_MONTHS    = {12, 1, 2, 3}      # best tourism months
MONSOON_MONTHS = {5, 6, 9, 10, 11}  # west/south coast affected

# Best place per purpose
BEST_PLACE = {
    "Beach":      "Mirissa",     "Adventure":  "Ella",
    "Cultural":   "Sigiriya",    "Eco-Tourism":"Wilpattu",
    "Honeymoon":  "Bentota",     "Business":   "Colombo",
    "Leisure":    "Galle",    
    "Medical":    "Colombo",     "Transit":    "Colombo",
    "Group Tour": "Kandy",       "Shopping":   "Colombo",
}
# Best month per purpose
BEST_MONTH = {
    "Beach":"December", "Adventure":"January", "Cultural":"February",
    "Eco-Tourism":"July","Honeymoon":"December","Business":"March",
    "Leisure":"January","VFR":"December","Medical":"February",
    "Transit":"December","Group Tour":"January","Shopping":"December",
}
# Friendly district -> place mapping
PLACE_LABELS = {
    "Colombo":"Colombo City","Gampaha":"Gampaha","Galle":"Galle Fort",
    "Mirissa":"Mirissa Beach","Bentota":"Bentota Beach","Hikkaduwa":"Hikkaduwa Beach",
    "Arugam Bay":"Arugam Bay","Ella":"Ella","Nuwara Eliya":"Nuwara Eliya",
    "Kandy":"Kandy","Sigiriya":"Sigiriya Rock","Anuradhapura":"Anuradhapura",
    "Polonnaruwa":"Polonnaruwa","Sinharaja":"Sinharaja Forest","Wilpattu":"Wilpattu National Park",
    "Knuckles Range":"Knuckles Range","Yala":"Yala National Park",
}
# Friendly purpose labels
PURPOSE_LABELS = {
    "VFR":     "Visiting Family/Friends"
}
PURPOSE_REV = {v: k for k, v in PURPOSE_LABELS.items()}

def encode(col, val):
    le = le_dict[col]
    return int(le.transform([val])[0]) if val in le.classes_ else 0

def predict(year, month_num, country, purpose, district, accom, crisis):
    df = pd.DataFrame([{
        "Year": year, "Month": month_num,
        "Country_of_Origin": encode("Country_of_Origin", country),
        "Purpose":           encode("Purpose", purpose),
        "Primary_District":  encode("Primary_District", district),
        "Accommodation_Type":encode("Accommodation_Type", accom),
        "Duration_Days": 7.0, "Group_Size": 2, "Age": 35.0,
        "Prior_Visits_SL": 0,
        "Season": encode("Season", "Yala" if month_num in [5,6,7,8,9] else "Maha"),
        "Crisis_Period": crisis,
    }])
    return max(0, int(model.predict(df)[0]))

def feat_row(label, status, your_val, best_val, note=""):
    cls  = {"good":"feat-good","warn":"feat-warn","bad":"feat-bad"}[status]
    tag  = {"good":"tag-good", "warn":"tag-warn", "bad":"tag-bad"}[status]
    icon = {"good":"✅","warn":"⚠️","bad":"❌"}[status]
    best_txt = f"<span style='color:#666;font-size:0.85rem;'> → Best: <b>{best_val}</b></span>" if best_val else ""
    note_txt = f"<br><span style='font-size:0.82rem;color:#555;'>{note}</span>" if note else ""
    st.markdown(f"""
    <div class='feat-row {cls}'>
        <span>{icon} <b>{label}</b>: {your_val}{best_txt}{note_txt}</span>
        <span class='{tag}'>{status.upper()}</span>
    </div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;color:#1a73e8;margin-bottom:0'>🌴 Sri Lanka Visit Planner</h1>", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown("<p style='color:#555;margin-bottom:0.5rem'>📝 Please add your details for the visit you are planning below.</p>", unsafe_allow_html=True)
with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        year    = st.selectbox("📅 Year",  FUTURE_YEARS,
                               key="sel_year",
                               index=None, placeholder="Select year…")
        month   = st.selectbox("🗓️ Month", MONTHS,
                               key="sel_month",
                               index=None, placeholder="Select month…")
        country = st.selectbox("🌍 Country",
                               sorted(le_dict["Country_of_Origin"].classes_),
                               key="sel_country",
                               index=None, placeholder="Select country…")
    with c2:
        EXCLUDE_PURPOSES = {"Transit", "VFR"}
        all_purposes = [p for p in sorted(le_dict["Purpose"].classes_) if p not in EXCLUDE_PURPOSES]
        purpose_opts = [PURPOSE_LABELS.get(p, p) for p in all_purposes]
        purpose_sel  = st.selectbox("🎯 Purpose",
                                    purpose_opts,
                                    key="sel_purpose",
                                    index=None, placeholder="Select purpose…")
        # Load ALL places directly from raw dataset
        all_districts = sorted(raw_df["Primary_District"].dropna().unique())
        place_opts    = [PLACE_LABELS.get(p, p) for p in all_districts]
        place_sel     = st.selectbox("📍 Place to Visit",
                                     place_opts,
                                     key="sel_place",
                                     index=None, placeholder="Select place…")
        accom = st.selectbox("🏨 Accommodation",
                             sorted(le_dict["Accommodation_Type"].classes_),
                             key="sel_accom",
                             index=None, placeholder="Select accommodation…")

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        go    = st.form_submit_button("🔍 Check My Visit", type="primary",    width="stretch")
    with btn_c2:
        clear = st.form_submit_button("🗑️ Clear",          type="secondary",  width="stretch")

# ── Clear: reset all form fields ────────────────────────────────────────────────────
if clear:
    for key in ["sel_year","sel_month","sel_country","sel_purpose","sel_place","sel_accom"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ── Result ────────────────────────────────────────────────────────────────────
if go:
    if not all([year, month, country, purpose_sel, place_sel, accom]):
        st.warning("⚠️ Please fill in **all fields** to continue.")
        st.stop()


    # Reverse map friendly labels -> raw values
    purpose  = PURPOSE_REV.get(purpose_sel, purpose_sel)
    rev_map  = {v: k for k, v in PLACE_LABELS.items()}
    district = rev_map.get(place_sel, place_sel)

    month_num = MONTHS.index(month) + 1
    is_crisis = 1 if (year, month_num) in CRISIS_PERIODS else 0
    season    = "Yala" if month_num in [5,6,7,8,9] else "Maha"

    pred = predict(year, month_num, country, purpose, district, accom, is_crisis)

    # ── Verdict card ──────────────────────────────────────────────────────────
    if is_crisis:
        em,col,title = "⚠️","#FF6D00","Avoid This Period"
        body = f"Only **{pred:,}** arrivals expected. Historical crisis period — very limited tourism."
    elif pred >= 160_000:
        em,col,title = "🌟","#34A853","Excellent Time to Visit!"
        body = f"**{pred:,}** arrivals expected — peak season with perfect conditions."
    elif pred >= 100_000:
        em,col,title = "✅","#4285F4","Good Time to Visit"
        body = f"**{pred:,}** arrivals expected — shoulder season, great value with fewer crowds."
    elif pred >= 50_000:
        em,col,title = "🟡","#FBBC04","Decent — But Not Ideal"
        body = f"**{pred:,}** arrivals expected — off-peak, some limitations apply."
    else:
        em,col,title = "❌","#EA4335","Not a Good Time"
        body = f"Only **{pred:,}** arrivals expected — very low season with significant drawbacks."

    st.markdown(f"""
    <div class='verdict-card' style='border-color:{col};background:{col}12;'>
        <h2 style='color:{col};margin:0'>{em} {title}</h2>
    </div>""", unsafe_allow_html=True)

    # ── Feature analysis ──────────────────────────────────────────────────────
    st.markdown("<div class='section-head'>🔎 Your Selection — What's Good & What's Not</div>",
                unsafe_allow_html=True)

    # Month
    best_m = BEST_MONTH.get(purpose, "December")
    if month_num in PEAK_MONTHS:
        feat_row("Month", "good",  month, None, "Peak tourism season — great choice!")
    elif month_num in MONSOON_MONTHS:
        feat_row("Month", "bad",   month, best_m,
                 "Monsoon can affect west/south coasts. Expect rain & rough seas.")
    else:
        feat_row("Month", "warn",  month, best_m, "Shoulder season — decent but not peak.")

    # Country
    country_data = raw_df[raw_df["Country_of_Origin"] == country]
    country_month = country_data[country_data["Month"]==month_num]["Monthly_Arrivals"].mean()
    country_avg   = country_data["Monthly_Arrivals"].mean()
    if country_month >= country_avg * 1.1:
        feat_row("Your Country", "good",  country, None, "Tourists from your country commonly visit in this month.")
    elif country_month < country_avg * 0.75:
        top_m = country_data.groupby("Month")["Monthly_Arrivals"].mean().idxmax()
        feat_row("Your Country", "bad",   country, f"Better in {MONTHS[top_m-1]}",
                 f"Tourists from {country} rarely travel to SL in {month}.")
    else:
        feat_row("Your Country", "warn",  country, None, "Average travel month for your country.")

    # Place
    best_place = BEST_PLACE.get(purpose, "Colombo")
    place_for_purpose = (district == best_place or
                         district in ["Colombo","Kandy","Galle","Sigiriya","Ella"])
    if district == best_place:
        feat_row("Place", "good", place_sel, None, f"Perfect match for {purpose} travel!")
    elif season == "Maha" and district in ["Mirissa","Hikkaduwa","Bentota","Arugam Bay"]:
        feat_row("Place", "bad",  place_sel, PLACE_LABELS.get(best_place, best_place),
                 "Beach areas hit by monsoon rains in wet season — waters can be rough.")
    else:
        feat_row("Place", "warn", place_sel, PLACE_LABELS.get(best_place, best_place),
                 f"For {purpose}, {PLACE_LABELS.get(best_place, best_place)} is usually the top pick.")

    # Accommodation
    budget_accom = accom in ["Budget Hotel","Hostel"]
    luxury_accom = accom in ["5-Star","Boutique"]
    if month_num in PEAK_MONTHS and budget_accom:
        feat_row("Accommodation", "warn", accom, "3-Star or 4-Star",
                 "Budget options book up fast in peak season — reserve early!")
    elif month_num in PEAK_MONTHS and luxury_accom:
        feat_row("Accommodation", "good", accom, None,
                 "Luxury stays are fully operational in peak season — great pick.")
    elif not month_num in PEAK_MONTHS and luxury_accom:
        feat_row("Accommodation", "warn", accom, "Boutique or Eco-Lodge",
                 "Off-peak rates are available — you may find better deals.")
    else:
        feat_row("Accommodation", "good", accom, None, "Good choice for your selected period.")

    # Season
    if season == "Yala":
        feat_row("Season", "good", "Dry Season (May–Sep)", None,
                 "Dry conditions — ideal for wildlife, east coast beaches & hill country.")
    else:
        feat_row("Season", "warn", "Wet Season (Oct–Apr)", "Dry Season",
                 "Wet season on west/south coast. Cultural Triangle stays dry & accessible.")

    # Crisis flag
    if is_crisis:
        feat_row("Safety", "bad", f"{year}-{month}", "Any post-2022 month",
                 "Historical crisis period — arrival numbers collapsed. Avoid if possible.")
    else:
        feat_row("Safety", "good", "No crisis flags", None,
                 "No historical crisis events for this period.")

    # ── Best combination suggestion ────────────────────────────────────────────
    st.markdown("<div class='section-head'>💡 Recommended Best Combination for You</div>",
                unsafe_allow_html=True)
    best_p  = PLACE_LABELS.get(BEST_PLACE.get(purpose,"Colombo"), BEST_PLACE.get(purpose,"Colombo"))
    best_mo = BEST_MONTH.get(purpose, "December")
    st.success(f"🗓️ **Best Month** for {purpose} in Sri Lanka → **{best_mo}**")
    st.success(f"📍 **Best Place** for {purpose} → **{best_p}**")
    st.success(f"☀️ **Best Season** → **Dry Season** (December – March for most coasts)")

    # ── Trend mini-chart ───────────────────────────────────────────────────────
    
    st.info("💡 Get an idea about your country's travel patterns to Sri Lanka and plan your visit at the perfect time!")


    st.markdown(f"<div class='section-head'>📈 Monthly Arrival Trend — {country}</div>",
                unsafe_allow_html=True)
    cdata = raw_df[raw_df["Country_of_Origin"] == country]
    if len(cdata) > 0:
        mdata = cdata.groupby("Month")["Monthly_Arrivals"].mean().reindex(range(1,13), fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 3))
        bar_colors = ["#1a73e8" if i+1 == month_num else
                      ("#34A853" if mdata[i+1] >= mdata.mean() else "#cccccc")
                      for i in range(12)]
        ax.bar([m[:3] for m in MONTHS], mdata.values, color=bar_colors, edgecolor="white")
        ax.axhline(mdata.mean(), color="#EA4335", linestyle="--", linewidth=1.2,
                   label=f"Avg: {mdata.mean():,.0f}")
        ax.set_ylabel("Avg Arrivals", fontsize=9)

        ax.set_title(f"When do tourists from {country} visit Sri Lanka?", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
        plt.xticks(fontsize=8); plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption(f"🔵 Blue bar = your selected month ({month})  |  🟢 Green = above-average months")

