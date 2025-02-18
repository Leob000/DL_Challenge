# %%
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import utils
from scipy.spatial import distance

#!%matplotlib inline
OPTION_FULL_ANALYSIS = False

# %%
df = pd.read_parquet("data/meteo.parquet")
df.set_index("date", inplace=True)
df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Paris")
df = df.sort_index()
# %%
df.columns
# %%
# Détails de la localisation de chaque station
# 1 station max par département
# Au moins 1 station dans chaque région
geo_details = df.groupby(["nom_reg", "nom_dept"])["numer_sta"].value_counts()
geo_details
# %%
# Affichage sur carte des stations et villes à pred
# On remarque que toutes les villes ont une station proche, sauf Grenoble
# On pourra donc pour la construction du modèle lier les données météo de chaque station
# à la ville correspondante, sauf pour Grenoble pour laquelle on peut faire la moyenne
# des trois stations les plus proches, plus ou moins équidistantes

# On prends les longitudes/latitudes des stations
df[["longitude", "latitude"]].round(9).value_counts()
location_counts = df[["longitude", "latitude"]].round(9).value_counts()
longi_stations = location_counts.index.get_level_values("longitude").tolist()
lat_stations = location_counts.index.get_level_values("latitude").tolist()
tuple_stations = (longi_stations, lat_stations)

# Idem pour les villes
villes_loc = pd.read_csv("data/villes_loc.csv")
tuple_villes = (villes_loc["Longitude"], villes_loc["Latitude"], villes_loc["Ville"])

# Voir la fonction pour afficher sur la carte dans utils.py
utils.plot_france_map(tuple_stations, tuple_villes)

# %%
# Distance euclidienne pour chaque ville aux stations, on trouve la station la
# plus proche (1 par département), pour Grenoble les 3 plus proches
distances = [
    (
        ville["Ville"],
        (lat, long),
        distance.euclidean((ville["Latitude"], ville["Longitude"]), (lat, long)),
        df.loc[(df["latitude"] == lat) & (df["longitude"] == long), "nom_dept"].values[
            0
        ],
    )
    for _, ville in villes_loc.iterrows()
    for lat, long in zip(lat_stations, longi_stations)
]

closest_stations = {
    ville: min([d for d in distances if d[0] == ville], key=lambda x: x[2])
    for ville in villes_loc["Ville"].unique()
}

grenoble_distances = [d for d in distances if d[0] == "Grenoble"]
closest_stations["Grenoble"] = sorted(grenoble_distances, key=lambda x: x[2])[:3]

for ville, station_info in closest_stations.items():
    if ville == "Grenoble":
        for station in station_info:
            print(ville, ":", station[3])
    else:
        print(ville, ":", station_info[3])

# %%
# On peut étudier le nombre de NaN pour chaque feature
pd.set_option("display.max_rows", 500)
df.isna().mean().sort_index()
# On met nbas en format numérique
df["nbas"] = pd.to_numeric(df["nbas"])
# %%
# On choisit ici les variables à garder, on va tester d'abord des modèles avec les variables essentielles
# Les variables essentielles:
cols_essential = [
    "ff",  # vitesse vent 10mn
    "tc",  # température celcius, pas besoin de garder les dérivés (min,max) de temp car bcp de NaN et déduisibles de tc
    "u",  # humidité
]
# Les variables potentiellement utiles:
cols_doubt = [
    "dd",  # direction du vent 10mn
    "temps_present",  # descri temps, idem que ww
    "n",  # utile mais bcp de NaN
    "nbas",  # idem, moins de NaN que n
    "ht_neige",  # garder?
    "rr1",  # garder? précipitations dans la dernière heure, passer en 30min?
    "raf10",  # rafales sur les 10mn, très corrélé avec ff, avec un peu plus de NaN
    "rafper",  # rafales sur la période? du coup avoir per aussi?
]
cols_geo = [
    "altitude",
    # "latitude",
    # "longitude",
    "nom_dept",
    "nom_reg",
]
df = df[cols_essential + cols_geo]
df.isna().mean().sort_values()
# %%
df["altitude"].value_counts().sort_index()
# %%
# On étudie comment sont répartis les NaN
msno.matrix(df.loc[:, df.isna().mean() > 0])
# df[cols_to_keep].corr().style.background_gradient(cmap="coolwarm").format(precision=2)
# %%
# Vent, temp et humidité par département
df.groupby(["nom_reg", "nom_dept"])[cols_essential].mean()
# %%
# Nombre de NaN par variable, par département
df.groupby(["nom_reg", "nom_dept"])[cols_essential].apply(lambda x: x.isna().sum())

# %%
# Bcp de NaN pour le Var
# On l'élimine car on a pas les valeurs pour le test set
depts = df["nom_dept"].unique().tolist()
if OPTION_FULL_ANALYSIS:
    for dept in depts:
        print(dept)
        msno.matrix(df[df["nom_dept"] == dept])
        plt.show()

df = df[df["nom_dept"] != "Var"]
df.groupby(["nom_dept"])[cols_essential].mean()
# %%
# Setting up multiindex
new_index = pd.date_range(
    start="2017-02-13 01:30:00+01:00",
    end="2022-12-31 23:30:00+01:00",
    freq="30min",
).tz_convert("Europe/Paris")
new_index

depts = df["nom_dept"].unique().tolist()
df_list = []
for dept in depts:
    df_temp = df[df["nom_dept"] == dept].copy()
    df_temp = df_temp.reindex(new_index)
    for col in ["nom_dept", "nom_reg", "altitude"]:
        df_temp[col] = df_temp[col].ffill().bfill()
    for col in ["ff", "tc", "u"]:
        df_temp[col] = df_temp[col].interpolate(method="time", limit_direction="both")
    df_list.append(df_temp)
df_list
df = pd.concat(df_list)
# %%
