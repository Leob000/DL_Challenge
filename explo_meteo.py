# %%
import matplotlib.pyplot as plt
import pandas as pd
import utils
from scipy.spatial import distance

# %%
df = pd.read_parquet("data/meteo.parquet")
df.set_index("date", inplace=True)
df.index = pd.to_datetime(df.index, utc=True)

# %%
# On drop les colones avec beaucoup de NaN
print(df.isna().mean().sort_values(ascending=False))
df = df.drop(columns=df.columns[df.isna().mean() > 0.5].tolist())

# %%
df.columns
# %%
# Drop les colonnes géo redondantes; on garde le nom_dept, nom_region
# (redondant? mais utile pour le join?), longi/lati (utile pour les villes?)
# Pas mal de noms epci mais au final c'est juste des doublons formattés différemment
df = df.drop(
  columns=[
    # "numer_sta",
    "coordonnees",
    "nom",
    "libgeo",
    "codegeo",
    "code_epci",
    "code_dep",
    "code_reg",
    "nom_epci",
  ]
)
# Drop colonnes inutiles
df = df.drop(columns=["mois_de_l_annee", "t", "tminsol"])
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
# des trois stations

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
# TODO: distance euclidienne pour chaque ville aux stations, trouver la station la
# plus proche (département), pour Grenoble les 3 plus proches
# Calculate the Euclidean distance between each city and each station
distances = []
for _, ville in villes_loc.iterrows():
  ville_coords = (ville["Latitude"], ville["Longitude"])
  for lat, long in zip(lat_stations, longi_stations):
    station_coords = (lat, long)
    dist = distance.euclidean(ville_coords, station_coords)
    nom_dept = df.loc[
      (df["latitude"] == lat) & (df["longitude"] == long), "nom_dept"
    ].values[0]
    distances.append((ville["Ville"], station_coords, dist, nom_dept))

# Find the closest station for each city
closest_stations = {}
for ville in villes_loc["Ville"].unique():
  ville_distances = [d for d in distances if d[0] == ville]
  closest_station = min(ville_distances, key=lambda x: x[2])
  closest_stations[ville] = closest_station

# Special case for Grenoble: find the three closest stations
grenoble_distances = [d for d in distances if d[0] == "Grenoble"]
three_closest_stations = sorted(grenoble_distances, key=lambda x: x[2])[:3]
closest_stations["Grenoble"] = three_closest_stations

closest_stations["Paris"][3]
for ville, station_info in closest_stations.items():
  if ville == "Grenoble":
    for station in station_info:
      print(ville, ":", station[3])
  else:
    print(ville, ":", station_info[3])
# %%
# S'occuper de temps_present / passe et ww / w1 / w2
# Vraiment pas bcp de w1, w2 donc tout drop sauf temps présent
col = ["ww", "w1", "w2", "temps_passe_1"]
df.drop(columns=col, inplace=True)
df.head()

# %%
lat_stations = df["latitude"].unique()
longi_stations = df["longitude"].unique()
lat_stations.shape, longi_stations.shape
sorted(longi_stations)

# %%
df.loc[df["longitude"] == "1.396666999999999", "longitude"] = "1.396667"
df.loc[df["longitude"] == "3.7640000000000002", "longitude"] = "3.764"
lat_stations = df["latitude"].unique()
longi_stations = df["longitude"].unique()
print(lat_stations.shape, longi_stations.shape)
print(sorted(longi_stations))

# %%
df.loc[df["longitude"] == "1.396667"]

# %%
print(sorted(longi_stations))

# %%

# Example longitude and latitude data
longi_stations = [-73.935242, -118.243683, -0.127758]
lat_stations = [40.730610, 34.052235, 51.507351]

plt.figure(figsize=(8, 6))
plt.scatter(longi_stations, lat_stations, color="red")
plt.title("Scatter Plot of Longitude and Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
