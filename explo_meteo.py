# %%
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry

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
# On prends les longitudes/latitudes des stations
df[["longitude", "latitude"]].round(9).value_counts()
location_counts = df[["longitude", "latitude"]].round(9).value_counts()
longi_stations = location_counts.index.get_level_values("longitude").tolist()
lat_stations = location_counts.index.get_level_values("latitude").tolist()

# Idem pour les villes
villes_loc = pd.read_csv("data/villes_loc.csv")
longi_villes = villes_loc["Longitude"]
lat_villes = villes_loc["Latitude"]


# Transformation pour affichage
def gdf_transfo(longi, lat):
  gdf = pd.DataFrame()
  gdf["longitudes"] = longi
  gdf["latitudes"] = lat
  gdf["geometry"] = gpd.points_from_xy(gdf["longitudes"], gdf["latitudes"])
  gdf = gpd.GeoDataFrame(gdf, crs="epsg:4326").to_crs("EPSG:3857")
  return gdf


gdf_stations = gdf_transfo(longi_stations, lat_stations)
gdf_villes = gdf_transfo(longi_villes, lat_villes)

url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

france = world[world["ADMIN"] == "France"]
france = (
  france["geometry"]
  .apply(
    lambda mp: shapely.geometry.MultiPolygon([p for p in mp.geoms if p.bounds[1] > 20])
  )
  .to_crs("EPSG:3857")
)
fig, ax = plt.subplots(figsize=(4, 5))

# Plot le pays
ax = france.boundary.plot(color="black", linewidth=0.5, alpha=0, ax=ax)
part1 = shapely.geometry.LineString(gdf_stations["geometry"].values)
linegdf = gpd.GeoDataFrame({"geometry": [part1]})

# Plot les stations
gdf_stations.plot(
  ax=ax,
  markersize=15,
  edgecolor="black",
  linewidth=0.5,
  zorder=1000,
)
# Plot les villes
gdf_villes.plot(
  ax=ax,
  markersize=10,
  color="red",
  edgecolor="black",
  linewidth=0.5,
  zorder=1001,  # force the points to be the top layer of the plot
)
ctx.add_basemap(ax=ax)
ax.set_axis_off()
plt.show()
# %%
# Load the map of France
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
france = world[world.name == "France"]

# Create a GeoDataFrame with the longitude and latitude values
gdf_stations = gpd.GeoDataFrame(
  geometry=gpd.points_from_xy(longi_stations, lat_stations)
)

# Plot the map of France
fig, ax = plt.subplots(figsize=(10, 10))
france.plot(ax=ax, color="white", edgecolor="black")
gdf_stations.plot(ax=ax, color="red", markersize=5)
plt.title("Scatter Plot of Longitude and Latitude on Map of France")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# %%
# longi = [-73.935242, -118.243683, -0.127758]
# lat = [40.730610, 34.052235, 51.507351]

# plt.figure(figsize=(8, 6))
plt.scatter(longi_stations, lat_stations, color="red")
plt.title("Scatter Plot of Longitude and Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

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
