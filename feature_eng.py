# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import holidays

GRAPHS = False  # On affiche les graphs ou non

STANDARDIZATION_PER_ZONE = True
DROP_AUGUSTS_FLAGS = True
DROP_PRECIPITATIONS = True
DROP_PRESSION = True

# %%
df_geo = pd.read_parquet("data/geo_tweaked.parquet")
df_meteo = pd.read_parquet("data/meteo_tweaked.parquet")
# %%
# On stocke les noms de regions et villes pour plus tard
regions = df_meteo.loc[df_meteo["is_reg"] == 1]["zone"].unique().tolist()
villes = df_meteo.loc[df_meteo["is_ville"] == 1]["zone"].unique().tolist()

df_meteo = df_meteo.drop(columns=["is_pays", "is_reg", "is_ville"])
df_meteo = df_meteo.reset_index(names="date")
# %%
# On transforme df_geo en version long
df2 = df_geo.reset_index(names="date")
df2 = pd.melt(df2, id_vars=["date"], var_name="zone", value_name="Load")
df_geo = df2.copy()

# %%
# Merge des deux
df = pd.merge(df_geo, df_meteo, on=["date", "zone"])

# %%
# On doit drop l'heure de changement d'heure octobre (présent dans l'index mais NaN pour geo, valeurs présentes pour meteo)
print(df.shape)
df.loc[(df["Load"].isna()) & (df["date"] < "2022")]
problem_dates = [
    "2017-10-29 02:00:00+02:00",
    "2017-10-29 02:30:00+02:00",
    "2018-10-28 02:00:00+02:00",
    "2018-10-28 02:30:00+02:00",
    "2019-10-27 02:00:00+02:00",
    "2019-10-27 02:30:00+02:00",
    "2020-10-25 02:00:00+02:00",
    "2020-10-25 02:30:00+02:00",
    "2021-10-31 02:00:00+02:00",
    "2021-10-31 02:30:00+02:00",
    "2022-10-30 02:00:00+02:00",
    "2022-10-30 02:30:00+02:00",
]
for dte in problem_dates:
    df.drop(df.loc[df["date"] == dte].index, inplace=True)
print(df.shape)
# df.loc[(df["Load"].isna()) & (df["date"] < "2022")] # Visu

# %%
# On drop les rows qui n'ont pas de valeur de conso
df_no_NaN = df[df["date"] < "2022"].dropna()
df = pd.concat([df_no_NaN, df[df["date"] >= "2022"]])
df.isna().mean()

# %%
# On recréé les flags indicateurs de pays, région, ville
for ville in villes:
    df.loc[df["zone"] == ville, "is_ville"] = 1
df["is_ville"] = df["is_ville"].fillna(0)

for region in regions:
    df.loc[df["zone"] == region, "is_reg"] = 1

df.loc[df["zone"] == "France", "is_pays"] = 1

df["is_ville"] = df["is_ville"].fillna(0)
df["is_reg"] = df["is_reg"].fillna(0)
df["is_pays"] = df["is_pays"].fillna(0)

# %%
# Feature eng: Température
# Temp flag <15 degrees celcius
# Exponential smoothin, alpha=0.15/0.06, min/max 24h of exp smoothing 0.15
df2 = df.copy()
df2 = df2.set_index("date")  # On met la date en index
df2["temp_below_15"] = df2["tc"] < 15

zones = df2["zone"].unique().tolist()

for zon in zones:
    df2.loc[df2["zone"] == zon, "tc_ewm15"] = (
        df2.loc[df2["zone"] == zon, "tc"].ewm(alpha=0.15).mean()
    )
    df2.loc[df2["zone"] == zon, "tc_ewm06"] = (
        df2.loc[df2["zone"] == zon, "tc"].ewm(alpha=0.06).mean()
    )
    df2.loc[df2["zone"] == zon, "tc_ewm15_max24h"] = (
        df2.loc[df2["zone"] == zon, "tc_ewm15"].rolling("24h").max()
    )
    df2.loc[df2["zone"] == zon, "tc_ewm15_min24h"] = (
        df2.loc[df2["zone"] == zon, "tc_ewm15"].rolling("24h").min()
    )

if GRAPHS:
    for i in ["tc", "tc_ewm15", "tc_ewm06", "tc_ewm15_max24h", "tc_ewm15_min24h"]:
        plt.plot(
            df2.loc[(df2["zone"] == "Paris") & (df2.index <= "2017-02-18"), i],
            linewidth=1,
            alpha=0.5,
        )
    plt.show()
    plt.scatter(
        df2.loc[df2["zone"] == "Paris", "tc"],
        df2.loc[df2["zone"] == "Paris", "Load"],
        alpha=0.2,
        s=7,
    )
    plt.show()

df = df2.copy()
# %%
# Feature eng: Dates
df2 = df.copy()

# August and (July or August) flags
df2["is_august"] = df2.index.month == 8
df2["is_july_or_august"] = (df2.index.month == 8) | (df2.index.month == 7)

# Trigonometry circles
df2["minute_of_day"] = df2.index.hour * 60 + df2.index.minute
df2["sin_time"] = np.sin(2 * np.pi * df2["minute_of_day"] / 1440)
df2["cos_time"] = np.cos(2 * np.pi * df2["minute_of_day"] / 1440)

df2["dayofweek"] = df2.index.dayofweek  # Monday=0, Sunday=6
df2["sin_dayofweek"] = np.sin(2 * np.pi * df2["dayofweek"] / 7)
df2["cos_dayofweek"] = np.cos(2 * np.pi * df2["dayofweek"] / 7)

df2["dayofyear"] = df2.index.dayofyear
df2["sin_dayofyear"] = np.sin(2 * np.pi * df2["dayofyear"] / 365)
df2["cos_dayofyear"] = np.cos(2 * np.pi * df2["dayofyear"] / 365)

# Flag weekend
df2["is_weekend"] = (df2["dayofweek"] == 5) | (df2["dayofweek"] == 6)

# Flag winter hour
df2["winter_hour"] = df2.index.map(
    lambda dt: 1 if dt.utcoffset() == pd.Timedelta(hours=1) else 0
)

# Flag single holiday
years = df2.index.year.unique()
fr_holidays = holidays.France(years=years)
df2["is_holiday"] = df2.index.map(lambda x: x.date() in fr_holidays)

df2.drop(columns=["minute_of_day", "dayofweek", "dayofyear"], inplace=True)
df = df2.copy()
# %%
# On met toutes les variables catégorielles en bool
li_bool = ["is_ville", "is_reg", "is_pays", "winter_hour"]
for col in li_bool:
    # print(df2[col].value_counts(dropna=False))
    df[col] = df[col].astype(bool)
    # print(df2[col].value_counts(dropna=False))

# %%
# Feature selection, on drop ou non certaines features, puis on normalise (sauf "Load")
features_to_normalize = [
    "ff",
    "tc",
    "u",
    "tc_ewm15",
    "tc_ewm06",
    "tc_ewm15_max24h",
    "tc_ewm15_min24h",
]

if DROP_AUGUSTS_FLAGS:
    df = df.drop(columns=["is_august", "is_july_or_august"])

if DROP_PRECIPITATIONS:
    df = df.drop(columns=["rr1"])
else:
    features_to_normalize.append("rr1")

if DROP_PRESSION:
    df = df.drop(columns=["pres"])
else:
    features_to_normalize.append("pres")

li_zones = df["zone"].unique().tolist()

# Normalisation des variables, par zone ou non
if STANDARDIZATION_PER_ZONE:
    for zone in li_zones:
        for feature in features_to_normalize:
            df.loc[df["zone"] == zone, feature] = (
                df.loc[df["zone"] == zone, feature]
                - df.loc[df["zone"] == zone, feature].mean()
            ) / df.loc[df["zone"] == zone, feature].std()
else:
    for feature in features_to_normalize:
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

# %%
df.to_parquet("data/clean_data.parquet", engine="pyarrow")
