# %%
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import pandas as pd
#!%matplotlib inline

plt.rcParams["figure.figsize"] = [10, 5]
GRAPHS = False  # On affiche les graphiques ou non

# %%
df = pd.read_csv("data/train.csv", index_col="date")
df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Paris")

# %%
# On vérif index integrity, il manque des indices pour les dates de changement d'heure octobre, les ajouter ou non ? Pour l'instant ne pas y toucher
expected_index = pd.date_range(
    start=df.index.min(), end=df.index.max(), freq="30min", tz="Europe/Paris"
)
if df.index.equals(expected_index):
    print("The index is complete. No rows are missing.")
else:
    missing = expected_index.difference(df.index)
    print("Missing timestamps:", missing)

# %%
# Simplification des noms de colonnes
df = df.rename(
    columns={
        # "Auvergne-Rhône-Alpes": "ARA",
        # "Bourgogne-Franche-Comté": "BFC",
        # "Centre-Val de Loire": "CVL",
        # "Grand Est": "GE",
        # "Hauts-de-France": "HDF",
        # "Nouvelle-Aquitaine": "NA",
        # "Pays de la Loire": "PL",
        # "Provence-Alpes-Côte d'Azur": "PACA",
        # "Île-de-France": "IDF",
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
if GRAPHS:
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
if GRAPHS:
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
# if OPTION_FULL_ANALYSIS:
# df.corr().style.background_gradient(cmap="coolwarm").format(precision=2)
if GRAPHS:
    corr_matrix = df.corr().round(2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.show()

# %%
# On enlève les mauvaises données pour Nancy
if GRAPHS:
    df["Nancy"].plot()
    plt.axvline(pd.Timestamp("2020-01-01"), color="r", linestyle="--")
df = df.assign(Nancy=lambda x: x["Nancy"].where(x.index >= "2020-01-01", float("nan")))
if GRAPHS:
    df["Nancy"].plot()
# %%
# On traite les mauvaises données de Nice
if GRAPHS:
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
if GRAPHS:
    df["Nice"].plot()
    plt.show()
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
if GRAPHS:
    df[laville].plot()

df = df.pipe(
    lambda x: x.assign(
        **{
            ville: x[ville].where(x[ville] >= seuil, float("nan"))
            for ville, seuil in ville_seuil
        }
    )
)
if GRAPHS:
    plt.axhline(90, color="r")
    plt.show()
    df[laville].plot()
# %%
# Correction de certains outliers de Nice
if GRAPHS:
    df.loc[(df.index >= "2021-04") & (df.index <= "2021-11"), "Nice"].plot()
    plt.axhline(510)
df.loc[(df.index >= "2021-04") & (df.index <= "2021-11"), "Nice"] = df.loc[
    (df.index >= "2021-04") & (df.index <= "2021-11"), "Nice"
].where(df["Nice"] <= 510, float("nan"))
if GRAPHS:
    df.loc[(df.index >= "2021-04") & (df.index <= "2021-11"), "Nice"].plot()

# %%
# Le boxplot ne montre plus de valeurs abérantes
if GRAPHS:
    df[villes].boxplot(vert=False, fontsize=10, grid=False)

# %%
# Corrélation semble bonne maintenant
if GRAPHS:
    corr_matrix = df.corr().round(2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.show()

# %%
# Seasonal plot for every year
if GRAPHS:
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
if GRAPHS:
    msno.matrix(df)
    plt.show()
# %%
# On interpole les NaN mineurs
df = df.interpolate(method="time", limit_area="inside")
if GRAPHS:
    msno.matrix(df)
    plt.show()

# Les NaN majeurs (valeurs manquantes 2017 pour les villes, <2020 pour Nancy) ne sont pas interpolés, on trainera juste ces zones sans ces valeurs

# %%
# Ajout de l'index 2022 à la df
df_2022 = pd.read_csv("data/pred_template.csv", index_col="date")
df_2022 = df_2022["pred_France"]
df_2022.index = pd.to_datetime(df_2022.index, utc=True).tz_convert("Europe/Paris")

df_concat = pd.concat([df, df_2022])
df_concat = df_concat.drop(labels="pred_France", axis=1)
df = df_concat

# %%
# Prolongement de l'index jusqu'à fin 2022
extended_index = pd.date_range(
    start=df.index.min(),
    end=pd.Timestamp("2022-12-31 23:30:00+01:00", tz="Europe/Paris"),
    freq="30min",
)
df = df.reindex(extended_index)
df

# %%
# Enregistrement des données traitées
df.to_parquet("data/geo_tweaked.parquet", engine="pyarrow")

# %%
