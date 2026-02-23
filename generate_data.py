import pandas as pd
import numpy as np
import random, os

SEED, N_SAMPLES = 42, 1500
np.random.seed(SEED); random.seed(SEED)

ANNUAL_ARRIVALS = {
    2015:1_798_380, 2016:2_050_832, 2017:2_116_407,
    2018:2_333_796, 2019:1_913_702, 2020:507_704,
    2021:194_495,   2022:719_978,   2023:1_487_000,
}
COUNTRY_SHARES = {
    "India":20.5,"United Kingdom":10.2,"Germany":6.8,"Maldives":6.0,
    "China":5.5,"Australia":4.8,"France":4.2,"USA":3.9,"Russia":3.5,
    "Canada":2.8,"Japan":2.5,"Netherlands":2.1,"Switzerland":1.9,
    "South Korea":1.7,"Italy":1.5,"Other":21.1,
}
SEA_W = np.array([10.5,9.8,9.2,7.5,6.5,6.8,8.2,8.0,6.5,7.0,8.5,11.5])
SEA_W /= SEA_W.sum()
YR_W  = np.array(list(ANNUAL_ARRIVALS.values()),dtype=float); YR_W/=YR_W.sum()
YEARS = list(ANNUAL_ARRIVALS.keys())

CRISIS = {
    (2019,4):0.25,(2019,5):0.35,(2019,6):0.50,
    (2020,3):0.30,(2020,4):0.02,(2020,5):0.02,(2020,6):0.02,
    (2020,7):0.02,(2020,8):0.02,(2020,9):0.02,(2020,10):0.05,
    (2020,11):0.08,(2020,12):0.15,
    (2021,1):0.05,(2021,2):0.05,(2021,3):0.05,(2021,4):0.05,
    (2021,5):0.05,(2021,6):0.05,(2021,7):0.10,(2021,8):0.15,
    (2021,9):0.25,(2021,10):0.35,(2021,11):0.50,(2021,12):0.60,
}
PURPOSE_BY_COUNTRY = {
    "India":["Leisure","Business","VFR","Medical"],
    "United Kingdom":["Leisure","Cultural","Adventure","Honeymoon"],
    "Germany":["Adventure","Eco-Tourism","Leisure","Cultural"],
    "Maldives":["Transit","Business","Leisure"],
    "China":["Leisure","Group Tour","Shopping"],
    "Australia":["Adventure","Leisure","VFR"],
    "France":["Cultural","Eco-Tourism","Leisure"],
    "USA":["Leisure","Business","Cultural"],
    "Russia":["Leisure","Beach","Honeymoon"],
    "Canada":["Leisure","Adventure","Cultural"],
    "Japan":["Cultural","Eco-Tourism","Leisure"],
    "Netherlands":["Eco-Tourism","Cultural","Leisure"],
    "Switzerland":["Adventure","Eco-Tourism","Leisure"],
    "South Korea":["Cultural","Leisure","Group Tour"],
    "Italy":["Cultural","Leisure","Honeymoon"],
    "Other":["Leisure","Cultural","Adventure"],
}
DISTRICT_BY_PURPOSE = {
    "Leisure":["Colombo","Galle","Mirissa","Bentota"],
    "Beach":["Mirissa","Hikkaduwa","Bentota","Arugam Bay"],
    "Cultural":["Kandy","Sigiriya","Anuradhapura","Polonnaruwa"],
    "Adventure":["Ella","Nuwara Eliya","Knuckles Range","Sigiriya"],
    "Eco-Tourism":["Sinharaja","Knuckles Range","Yala","Wilpattu"],
    "Business":["Colombo","Gampaha"],
    "Honeymoon":["Bentota","Galle","Nuwara Eliya","Mirissa"],
    "VFR":["Colombo","Gampaha","Kandy"],
    "Medical":["Colombo"],"Transit":["Colombo"],
    "Group Tour":["Colombo","Kandy","Sigiriya","Galle"],
    "Shopping":["Colombo","Gampaha"],
}
ACCOMMODATION = ["Budget Hotel","3-Star","4-Star","5-Star","Boutique","Hostel","Eco-Lodge","Homestay"]
DURATION = {"Leisure":(5,14),"Beach":(4,12),"Cultural":(4,10),"Adventure":(7,18),
            "Eco-Tourism":(5,14),"Business":(2,6),"Honeymoon":(8,21),
            "VFR":(6,21),"Medical":(5,30),"Transit":(1,3),
            "Group Tour":(4,10),"Shopping":(2,7)}

def get_arrivals(year,month):
    base = ANNUAL_ARRIVALS[year]*SEA_W[month-1]
    return max(0,int(base*CRISIS.get((year,month),1.0)*np.random.normal(1.0,0.05)))

countries = list(COUNTRY_SHARES.keys())
shares = np.array([COUNTRY_SHARES[c] for c in countries],dtype=float); shares/=shares.sum()

rows=[]
for _ in range(N_SAMPLES):
    yr  = np.random.choice(YEARS,p=YR_W)
    mo  = np.random.choice(range(1,13),p=SEA_W)
    c   = np.random.choice(countries,p=shares)
    pur = random.choice(PURPOSE_BY_COUNTRY[c])
    dis = random.choice(DISTRICT_BY_PURPOSE.get(pur,["Colombo"]))
    acc = random.choice(ACCOMMODATION)
    lo,hi = DURATION.get(pur,(4,14))
    dur = int(np.random.uniform(lo,hi))
    gs  = int(np.random.choice([1,2,3,4,5,6],p=[0.30,0.35,0.15,0.10,0.06,0.04]))
    age = int(max(18,min(75,np.random.normal(38,12))))
    pv  = int(np.random.choice([0,1,2,3,4],p=[0.55,0.25,0.10,0.06,0.04]))
    # ← Updated season names
    season = "Dry Season" if mo in [5,6,7,8,9] else "Wet Season"
    crisis = 1 if CRISIS.get((yr,mo),1.0) < 0.5 else 0
    arr = get_arrivals(yr,mo)
    rows.append({"Year":yr,"Month":mo,"Country_of_Origin":c,"Purpose":pur,
                 "Primary_District":dis,"Accommodation_Type":acc,
                 "Duration_Days":dur,"Group_Size":gs,"Age":age,
                 "Prior_Visits_SL":pv,"Season":season,
                 "Crisis_Period":crisis,"Monthly_Arrivals":arr})

df = pd.DataFrame(rows)

# Introduce messiness
df.loc[df.sample(frac=0.08,random_state=1).index,"Duration_Days"] = np.nan
df.loc[df.sample(frac=0.06,random_state=2).index,"Age"]           = np.nan
typos={"India":"Indai","Germany":"Germny","United Kingdom":"United Kingdm",
       "Australia":"Austraila","France":"Frnace"}
for idx in df.sample(frac=0.04,random_state=3).index:
    orig=df.loc[idx,"Country_of_Origin"]
    if orig in typos: df.loc[idx,"Country_of_Origin"]=typos[orig]
df = pd.concat([df,df.sample(frac=0.02,random_state=4)],ignore_index=True)
df = df.sample(frac=1,random_state=SEED).reset_index(drop=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","sl_tourism_raw.csv")
df.to_csv(out,index=False)
print(f"Saved: {out}")
print(f"Shape: {df.shape}")
print(f"Unique seasons: {df['Season'].unique()}")
print(f"Missing:\n{df.isnull().sum()[df.isnull().sum()>0]}")
