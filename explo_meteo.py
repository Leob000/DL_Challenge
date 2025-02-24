# %%
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import utils
from scipy.spatial import distance

#!%matplotlib inline
FULL_ANALYSIS = False

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
if FULL_ANALYSIS:
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

if FULL_ANALYSIS:
    for ville, station_info in closest_stations.items():
        if ville == "Grenoble":
            for station in station_info:
                print(ville, ":", station[3])
        else:
            print(ville, ":", station_info[3])

# %%
# On met nbas en format numérique
df["nbas"] = pd.to_numeric(df["nbas"])

# On peut étudier le nombre de NaN pour chaque feature
pd.set_option("display.max_rows", 500)
df.isna().mean().sort_index()
# %%
# On choisit ici les variables à garder, on va tester d'abord des modèles avec les variables essentielles
# Les variables essentielles:
cols_essential = [
    "ff",  # vitesse vent 10mn
    "tc",  # température celcius, pas besoin de garder les dérivés (min,max) de temp car bcp de NaN et déduisibles de tc
    "u",  # humidité
    # "rr1",  # précipitations dans la dernière heure
    "pres",  # pression au niveau de la stations
]
# Les variables potentiellement utiles:
cols_doubt = [
    "dd",  # direction du vent 10mn
    "temps_present",  # descri temps, idem que ww
    "n",  # utile mais bcp de NaN
    "nbas",  # idem, moins de NaN que n
    "ht_neige",  # trop de NaN
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
# On étudie comment sont répartis les NaN
# On remarque sur la matrice de corrélation des variables choisies qu'elles ne semblent pas très corrélées, ce qui est plutôt bon, elles apportent de l'information différente
if FULL_ANALYSIS:
    msno.matrix(df.loc[:, df.isna().mean() > 0])
    plt.show()
    print(df[cols_essential].corr())
# %%
# Vent, temp et humidité par département
if FULL_ANALYSIS:
    df.groupby(["nom_reg", "nom_dept"])[cols_essential].mean()
# %%
# Nombre de NaN par variable, par département
if FULL_ANALYSIS:
    df.groupby(["nom_reg", "nom_dept"])[cols_essential].apply(lambda x: x.isna().sum())

# %%
# Bcp de NaN pour le Var
# On l'élimine car on a pas les valeurs pour le test set
depts = df["nom_dept"].unique().tolist()
if FULL_ANALYSIS:
    for dept in depts:
        print(dept)
        msno.matrix(df[df["nom_dept"] == dept])
        plt.show()

df = df[df["nom_dept"] != "Var"]

# %%
# On met le même index que train+test geo dataset, on inteprole pour passer d'intervalle 3h à 30min
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
    for col in [cols_essential]:
        df_temp[col] = df_temp[col].interpolate(method="time", limit_direction="both")
    df_list.append(df_temp)
df_list
df = pd.concat(df_list)
# %%
# On créé une nouvelle feature `France` qui regroupe la moyenne de toutes les stations
df2 = df.copy()
df2 = df2.drop(columns="altitude")
df_france = df2.groupby(df2.index)[cols_essential].mean()
df_france["zone"] = "France"
df_france["is_pays"] = 1
df2 = pd.concat([df2, df_france])
df2["is_pays"] = df2["is_pays"].fillna(0)

df = df2.copy()
df2["is_pays"].value_counts(dropna=False)

# %%
# On créé une nouvelle feature par région, faisant la moyenne valeurs des stations de cette région
df2 = df.copy()
regions = df2["nom_reg"].unique().tolist()
regions = [x for x in regions if x == x]  # Retire le nan
regions
df_regions = df2.groupby(["nom_reg", df2.index])[cols_essential].mean()
df_regions["zone"] = df_regions.index.get_level_values("nom_reg")

df_regions = df_regions.reset_index(level="nom_reg", drop=True)
df_regions["is_reg"] = 1
df2 = pd.concat([df2, df_regions])
df2["is_reg"] = df2["is_reg"].fillna(0)

df = df2.copy()
df2["is_reg"].value_counts(dropna=False)

# %%
# On associe les stations aux villes qui correspondent, en faisant une moyenne pour Grenoble
df2 = df.copy()
ville_dept = [
    ("Montpellier", "Hérault"),
    ("Lille", "Nord"),
    ("Nice", "Alpes-Maritimes"),
    ("Rennes", "Ille-et-Vilaine"),
    ("Rouen", "Seine-Maritime"),
    ("Marseille", "Bouches-du-Rhône"),
    ("Lyon", "Rhône"),
    ("Nancy", "Meurthe-et-Moselle"),
    ("Paris", "Essonne"),
    ("Nantes", "Loire-Atlantique"),
    ("Toulouse", "Haute-Garonne"),
]
for ville, dept in ville_dept:
    df2.loc[df2["nom_dept"] == dept, "zone"] = ville
    df2.loc[df2["nom_dept"] == dept, "is_ville"] = 1

# Moyenne des 2 stations Lyon, Hautes-Alphes pour Grenoble
df3 = df2.loc[(df2["nom_dept"] == "Rhône") | (df2["nom_dept"] == "Hautes-Alpes")]
df3 = df3.groupby(df3.index)[cols_essential].mean()
df3["zone"] = "Grenoble"
df3["is_ville"] = 1

df2 = pd.concat([df2, df3])
df2["is_ville"] = df2["is_ville"].fillna(0)

df = df2.copy()
df2["is_ville"].value_counts(dropna=False)

# %%
# On élimine les anciennes features pour chaque station,
# On ne garde que les observations soit pays, soit region, soit ville
li_is = ["is_pays", "is_reg", "is_ville"]
df["is_pays"] = df["is_pays"].fillna(0)
df["is_reg"] = df["is_reg"].fillna(0)
if FULL_ANALYSIS:
    for i in li_is:
        print(df[i].value_counts(dropna=False))

df = df[(df["is_pays"] == 1) | (df["is_reg"] == 1) | (df["is_ville"] == 1)]

if FULL_ANALYSIS:
    for i in li_is:
        print(df.loc[df[i] == 1]["zone"].value_counts(dropna=False))

df = df.drop(columns=["nom_dept", "nom_reg"])
# %%
# df[df["is_reg"] == 1]["zone"].value_counts()
df.to_parquet("data/meteo_tweaked.parquet", engine="pyarrow")

# %%
