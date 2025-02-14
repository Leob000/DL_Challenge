# %%
import missingno as msno
import pandas as pd
import utils
from scipy.spatial import distance

# %%
df = pd.read_parquet("data/meteo.parquet")
df.set_index("date", inplace=True)
df.index = pd.to_datetime(df.index, utc=True)


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
    df.loc[(df["latitude"] == lat) & (df["longitude"] == long), "nom_dept"].values[0],
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
nan = df.isna().mean().sort_index()
nan
# df = df.drop(columns=df.columns[df.isna().mean() > 0.5].tolist())

# %%
cols_to_keep = [
  "dd",  # direction du vent 10mn
  "ff",  # vitesse vent 10mn
  "tc",  # température celcius, pas besoin de garder les dérivés (min,max) de temp car bcp de NaN et déduisibles de tc
  "u",  # humidité
  "temps_present",  # descri temps
  "n",  # nébulosité, mais bcp de NaN
  "nbas",  # nébulosité basse, bcp moins de NaN que n
  "raf10",  # rafales sur les 10mn
]
cols_geo = [
  "altitude",  # ?
  "latitude",
  "longitude",
  "nom_dept",
  "nom_reg",
]
cols_doubt = [
  "ht_neige",  # garder?
  "rr1",  # garder? précipitations dans la dernière heure, passer en 30min?
  "rafper",  # rafales sur la période? du coup avoir per aussi?
]
df = df[cols_to_keep + cols_geo]
df.isna().mean().sort_values()
# %%
# n semble utile intuitivement mais 50% de NaN
# nbas aussi mais 19% NaN
# TODO Gérer ces NaN
msno.matrix(df.loc[:, df.isna().mean() > 0])
# %%

# %%
# TODO Créer fonction tweak_meteo
# TODO Voir la dernière cellule, check les colonnes de la df grader que les utiles
