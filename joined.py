# %%
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

OPTION_FULL_ANALYSIS = False

# %%
df_geo = pd.read_parquet("data/geo_tweaked.parquet")
df_meteo = pd.read_parquet("data/meteo_tweaked.parquet")
# %%
regions = df_meteo.loc[df_meteo["is_reg"] == 1]["zone"].unique().tolist()
villes = df_meteo.loc[df_meteo["is_ville"] == 1]["zone"].unique().tolist()

# %%
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
# msno.matrix(df) # Visu

# %%
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
# Feature eng, binning temp
# On met la date en index
df2 = df.copy()
df2 = df2.set_index("date")
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

if OPTION_FULL_ANALYSIS:
    for i in ["tc", "tc_ewm15", "tc_ewm06"]:
        plt.plot(
            df2.loc[(df2["zone"] == "Paris") & (df2.index <= "2017-02-18"), i],
            linewidth=1,
            alpha=0.5,
        )
    plt.show()

df = df2.copy()

# %%
