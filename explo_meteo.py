# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_parquet("data/meteo.parquet")
df.set_index("date", inplace=True)
df.index = pd.to_datetime(df.index, utc=True)

# %%
nan_percent = df.isna().mean().sort_values(ascending=False)

# %%
# On drop les features avec >90% de NaN, features peu utiles par ailleurs
df.drop(columns=nan_percent[nan_percent > 0.9].index, inplace=True)
df.head()

# %%
# Drop les colonnes géo redondantes; on garde le nom_dept, nom_region (redondant? mais utile pour le join?), longi/lati (utile pour les villes?)
# Pas mal de noms epci mais au final c'est juste des doublons formattés différemment
df.drop(
  columns=[
    "numer_sta",
    "coordonnees",
    "nom",
    "libgeo",
    "codegeo",
    "code_epci",
    "code_dep",
    "code_reg",
    "nom_epci",
  ],
  inplace=True,
)
# Drop var inutiles
df.drop(columns=["mois_de_l_annee", "t", "tminsol", "tn12", "tx12"], inplace=True)

# %%
new_order = [
  "date",
  "nom_dept",
  "nom_reg",
  "latitude",
  "longitude",
  "altitude",
  "pmer",
  "tend",
  "cod_tend",
  "dd",
  "ff",
  "td",
  "u",
  "vv",
  "n",
  "nbas",
  "hbas",
  "cl",
  "cm",
  "pres",
  "tend24",
  "raf10",
  "rafper",
  "per",
  "etat_sol",
  "ht_neige",
  "rr1",
  "rr3",
  "rr6",
  "rr12",
  "rr24",
  "nnuage1",
  "ctype1",
  "hnuage1",
  "nnuage2",
  "hnuage2",
  "nnuage3",
  "hnuage3",
  "type_de_tendance_barometrique",
  "temps_present",
  "ww",
  "temps_passe_1",
  "w1",
  "w2",
  "tc",
  "tn12c",
  "tx12c",
  "tminsolc",
]
df = df[new_order]

# %%
# S'occuper de temps_present / passe et ww / w1 / w2
# Vraiment pas bcp de w1, w2 donc tout drop sauf temps présent
col = ["ww", "w1", "w2", "temps_passe_1"]
df.drop(columns=col, inplace=True)
df.head()

# %%
lat = df["latitude"].unique()
longi = df["longitude"].unique()
lat.shape, longi.shape
sorted(longi)

# %%
df.loc[df["longitude"] == "1.396666999999999", "longitude"] = "1.396667"
df.loc[df["longitude"] == "3.7640000000000002", "longitude"] = "3.764"
lat = df["latitude"].unique()
longi = df["longitude"].unique()
print(lat.shape, longi.shape)
print(sorted(longi))

# %%
df.loc[df["longitude"] == "1.396667"]

# %%
print(sorted(longi))

# %%

# Example longitude and latitude data
longi = [-73.935242, -118.243683, -0.127758]
lat = [40.730610, 34.052235, 51.507351]

plt.figure(figsize=(8, 6))
plt.scatter(longi, lat, color="red")
plt.title("Scatter Plot of Longitude and Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
