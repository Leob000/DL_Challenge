# %%
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd

#!%matplotlib inline

plt.rcParams["figure.figsize"] = [10, 5]
OPTION_FULL_ANALYSIS = False  # Analyse complète ou non
OPTION_NICE_SHIFT = True  # Mauvaises données "Nice" shiftées ou éliminées

# %%
df = pd.read_csv("data/train.csv", index_col="date")
df.index = pd.to_datetime(df.index, utc=True)
# %%
# TODO clean ça
test2 = df.loc[(df.index >= "2018-10-27") & (df.index <= "2018-10-29"), :]
test2
# Add missing indices
missing_indices = pd.date_range(
  start="2018-10-28 00:00:00+00:00", end="2018-10-28 00:30:00+00:00", freq="30T"
)
test3 = test2.reindex(test2.index.union(missing_indices))
test3
# %%
# Simplification des noms de colonnes
df = df.rename(
  columns={
    "Auvergne-Rhône-Alpes": "ARA",
    "Bourgogne-Franche-Comté": "BFC",
    "Centre-Val de Loire": "CVL",
    "Grand Est": "GE",
    "Hauts-de-France": "HDF",
    "Nouvelle-Aquitaine": "NA",
    "Pays de la Loire": "PL",
    "Provence-Alpes-Côte d'Azur": "PACA",
    "Île-de-France": "IDF",
    "Montpellier Méditerranée Métropole": "Montpellier",
    "Métropole Européenne de Lille": "Lille",
    "Métropole Grenoble-Alpes-Métropole": "Grenoble",
    "Métropole Nice Côte d'Azur": "Nice",
    "Métropole Rennes Métropole": "Rennes",
    "Métropole Rouen Normandie": "Rouen",
    "Métropole d'Aix-Marseille-Provence": "Marseille",
    "Métropole de Lyon": "Lyon",
    "Métropole du Grand Nancy": "Nancy",
    "Métropole du Grand Paris": "Paris",
    "Nantes Métropole": "Nantes",
    "Toulouse Métropole": "Toulouse",
  },
)
regions = list(df.columns)[1:13]
villes = list(df.columns)[13:]
villes_no_paris = villes.copy()
villes_no_paris.remove("Paris")
regions_france = ["France"] + regions
cols = list(df.columns)

# %%
# Recherche valeurs abérante
if OPTION_FULL_ANALYSIS:
  df[regions_france].boxplot(
    vert=False,
    fontsize=10,
    grid=False,
  )
  plt.show()

  # Valeurs abérantes pour les villes, qu'on supprime plus tard
  df[villes].boxplot(vert=False, fontsize=10, grid=False)

# %%
# Visualisation histogrammes
if OPTION_FULL_ANALYSIS:
  df[villes].hist(
    bins=50,
  )
  plt.show()
  df[regions].hist(
    bins=50,
  )
  plt.show()
# %%
# Corrélation étrange pour Nancy, Nice
if OPTION_FULL_ANALYSIS:
  df.corr().style.background_gradient(cmap="coolwarm").format(precision=2)

# %%
# On enlève les mauvaises données pour Nancy
df["Nancy"].plot()
plt.axvline(pd.Timestamp("2020-01-01"), color="r", linestyle="--")
df = df.assign(Nancy=lambda x: x["Nancy"].where(x.index >= "2020-01-01", float("nan")))
df["Nancy"].plot()
# %%
# On traite les mauvaises données de Nice
if OPTION_NICE_SHIFT:  # Shifting à la main des mauvaises données
  df["Nice"].plot()
  plt.axvline(pd.Timestamp("2021-08-25"), color="r", linestyle="--")
  plt.show()
  dftest = df.copy()
  # dftest.loc[dftest.index >= "2021-08", "Nice"].plot()
  # plt.show()
  dftest = (
    df.assign(
      Nice=lambda x: x["Nice"].where(x.index < "2021-08-25 09:30:00", x["Nice"] + 150)
    )
    .assign(Nice=lambda x: x["Nice"].where(x.index < "2021-08-30", x["Nice"] - 150))
    .assign(Nice=lambda x: x["Nice"].where(x.index < "2021-09-03", x["Nice"] + 150))
  )
  # dftest.loc[dftest.index >= "2021-08", "Nice"].plot()
  # plt.show()
  df = dftest.copy()
  df["Nice"].plot()
  plt.show()
else:  # Elimination de ces données
  df["Nice"].plot()
  plt.axvline(pd.Timestamp("2021-08-01"), color="r", linestyle="--")
  df = df.assign(Nice=lambda x: x["Nice"].where(x.index <= "2021-08-01", float("nan")))
  df["Nice"].plot()
# %%
# Beaucoups de valeurs aberrantes pour les villes,
# on les cherche une par une puis les élimine
ville_seuil = [
  ("Montpellier", 130),
  ("Lille", 300),
  ("Grenoble", 245),
  ("Nice", 180),
  ("Rennes", 110),
  ("Rouen", 110),
  ("Marseille", 600),
  ("Lyon", 500),
  ("Paris", 500),
  ("Nantes", 90),
  ("Toulouse", 230),
]
laville = "Nice"
df[laville].plot()

df = df.pipe(
  lambda x: x.assign(
    **{
      ville: x[ville].where(x[ville] >= seuil, float("nan"))
      for ville, seuil in ville_seuil
    }
  )
)
plt.axhline(90)
plt.show()
df[laville].plot()
# %%
# Le boxplot ne montre plus de valeurs abérantes
if OPTION_FULL_ANALYSIS:
  df[villes].boxplot(vert=False, fontsize=10, grid=False)

# %%
# Corrélation semble bonne maintenant
if OPTION_FULL_ANALYSIS:
  df.corr().style.background_gradient(cmap="coolwarm").format(precision=2)

# %%
# Seasonal plot for every year
if OPTION_FULL_ANALYSIS:
  year_min = df.index.min().year
  year_max = df.index.max().year
  for year in range(year_min, year_max + 1):
    df_year = df.loc[f"{year}"]
    df_year.groupby(df_year.index.month).mean()["France"].plot(label=f"{year}")
  plt.legend()
  plt.title("Seasonal Plot for Every Year")
  plt.xlabel("Month")
  plt.ylabel("Average Value")
  plt.show()

# %%
# 2 types de valeurs manquantes, soit globalement communes à toutes les villes
# soit manque du début ~2017 différent pour chaque ville
msno.matrix(df)
plt.show()

# %%
test = df.loc[(df.index >= "2018-10-25") & (df.index <= "2018-10-30"), "Montpellier"]
test
# %%
# On remplace chaque NaN non consécutif avec la moyenne entre t-1 et t+1
city = "Montpellier"
nan_dates = df[df[city].isna()].index
non_consecutive_nan_dates = nan_dates[
  nan_dates.to_series().diff() != pd.Timedelta("30 minutes")
]
# On enlève le premier jour sinon bug
non_consecutive_nan_dates = non_consecutive_nan_dates[1:]
non_consecutive_nan_dates

for date in non_consecutive_nan_dates:
  before = df.loc[date - pd.DateOffset(minutes=30), city]
  after = df.loc[date + pd.DateOffset(minutes=30), city]
  df.at[date, city] = (before + after) / 2

msno.matrix(df[city])
plt.show()

# %%
# Check for non-consecutive NaN values for a specific city
city = "Montpellier"
nan_dates = df[df[city].isna()].index
non_consecutive_nan_dates = nan_dates[
  nan_dates.to_series().diff() != pd.Timedelta("30 minutes")
]
non_consecutive_nan_dates = non_consecutive_nan_dates[
  1:
]  # On enlève le premier jour du dataset
print(non_consecutive_nan_dates)

for date in non_consecutive_nan_dates:
  trente_min = pd.DateOffset(minutes=30)
  # print(type(date))
  # print(type(trente_min))
  # print(type(date - trente_min))
  before = df.loc[date - trente_min, city]
  after = df[city].loc[date + pd.DateOffset(minutes=30)]
  df.at[date, city] = (before + after) / 2

# %%
# Coupure jusque 2017-10-23 07:00:00+00:00 ? Pour toutes les villes
pd.DataFrame(df[(df.index.year) & (df["Montpellier"].isna())].index)

# %%
# Etude des NaN communs aux villes, sont ils exactement la même date? consécutifs?
villes_no_nancy = villes.copy()
villes_no_nancy.remove("Nancy")

# msno.matrix(df.loc[(df.index >= "2018") & (df.index <= "2019"), villes])
df.loc[(df.index >= "2018") & (df.index <= "2019"), villes].isna().mean()
# df[(df.index.year > 2017) & (df[villes_no_nancy].isna())].index

# %%
# Fonction pour appliquer toutes les transformations faites dans ce notebook


def geo_tweak(df):
  ville_seuil = [
    ("Montpellier", 130),
    ("Lille", 300),
    ("Grenoble", 245),
    ("Nice", 180),
    ("Rennes", 110),
    ("Rouen", 110),
    ("Marseille", 600),
    ("Lyon", 500),
    ("Paris", 500),
    ("Nantes", 90),
    ("Toulouse", 230),
  ]
  return (
    df.rename(
      columns={
        "Auvergne-Rhône-Alpes": "ARA",
        "Bourgogne-Franche-Comté": "BFC",
        "Centre-Val de Loire": "CVL",
        "Grand Est": "GE",
        "Hauts-de-France": "HDF",
        "Nouvelle-Aquitaine": "NA",
        "Pays de la Loire": "PL",
        "Provence-Alpes-Côte d'Azur": "PACA",
        "Île-de-France": "IDF",
        "Montpellier Méditerranée Métropole": "Montpellier",
        "Métropole Européenne de Lille": "Lille",
        "Métropole Grenoble-Alpes-Métropole": "Grenoble",
        "Métropole Nice Côte d'Azur": "Nice",
        "Métropole Rennes Métropole": "Rennes",
        "Métropole Rouen Normandie": "Rouen",
        "Métropole d'Aix-Marseille-Provence": "Marseille",
        "Métropole de Lyon": "Lyon",
        "Métropole du Grand Nancy": "Nancy",
        "Métropole du Grand Paris": "Paris",
        "Nantes Métropole": "Nantes",
        "Toulouse Métropole": "Toulouse",
      },
    )
    .assign(Nancy=lambda x: x["Nancy"].where(x.index >= "2020-01-01", float("nan")))
    .assign(
      Nice=lambda x: x["Nice"].where(x.index < "2021-08-25 09:30:00", x["Nice"] + 150)
    )
    .assign(Nice=lambda x: x["Nice"].where(x.index < "2021-08-30", x["Nice"] - 150))
    .assign(Nice=lambda x: x["Nice"].where(x.index < "2021-09-03", x["Nice"] + 150))
    .pipe(
      lambda x: x.assign(
        **{
          ville: x[ville].where(x[ville] >= seuil, float("nan"))
          for ville, seuil in ville_seuil
        }
      )
    )
  )
