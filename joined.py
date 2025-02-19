# %%
import pandas as pd
import missingno as msno

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
msno.matrix(df.sort_values(by="date"))
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
df.loc[(df["Load"].isna()) & (df["date"] < "2022")]

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
df
