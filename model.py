# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

plt.rcParams["figure.figsize"] = [10, 5]

COLAB = False  # Si utilisation de google colab
FULL_TRAIN = True  # True: pred sur 2022, False: pred sur 2021 (validation)
if COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    df = pd.read_parquet("drive/MyDrive/data/clean_data.parquet")
else:
    df = pd.read_parquet("data/clean_data.parquet")
# %%
# On normalise les variables continues, on choisit pour l'instant de normaliser par région
# On garde en liste les moyennes et std des Loads des différentes régions pour renormaliser les pred à la fin
li_zones = [col for col in df.columns if col.startswith("zone_")]
li_norm = [
    "Load",
    "ff",
    "tc",
    "u",
    "tc_ewm15",
    "tc_ewm06",
    "tc_ewm15_max24h",
    "tc_ewm15_min24h",
]

li_load = []
for zone in li_zones:
    for feature in li_norm:
        if feature == "Load":
            Load_mean = (zone, df.loc[df[zone] == 1, feature].mean())
            Load_sd = (zone, df.loc[df[zone] == 1, feature].std())
            load_temp = (
                zone,
                df.loc[df[zone] == 1, feature].mean(),
                df.loc[df[zone] == 1, feature].std(),
            )
            li_load.append(load_temp)
        df.loc[df[zone] == 1, feature] = (
            df.loc[df[zone] == 1, feature] - df.loc[df[zone] == 1, feature].mean()
        ) / df.loc[df[zone] == 1, feature].std()


# Fonction à utilier plus tard pour rescale
def rescale(dff, col):
    for zone, mean, std in li_load:
        dff.loc[dff[zone] == 1, col] = dff.loc[dff[zone] == 1, col] * std + mean


# %%
# Création des X/y train/test
if FULL_TRAIN:
    df_train = df[df.index < "2022"]
    df_test = df[df.index >= "2022"]
else:
    df_train = df[df.index < "2021"]
    df_test = df[(df.index >= "2021") & (df.index < "2022")]

X_train = df_train.drop(columns=["Load"]).to_numpy(dtype="float32")
y_train = df_train["Load"].to_numpy(dtype="float32")

X_test = df_test.drop(columns=["Load"]).to_numpy(dtype="float32")
y_test = df_test["Load"].to_numpy(dtype="float32")


# %%
# Importation du modèle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

# Seed 42 sauf si précisé
# VALIDATION TRAIN
# MLP (50)
# err_train: 4970.189566458095 err_val: 5305.83536908326
# MLP (100,75,50) alpha=0.0001 12m
# err_train: 3520.0179453667292 err_val: 5566.213886271534
# MLP (100,75,50) alpha=0.0005 12m
# err_train: 3562.9267325270466 err_val: 5432.463928511991
# MLP (100,75,50) alpha=0.001 17m colab
# err_train: 3572.5110659410075 err_val: 5384.728522402602
# MLP (150,75,50) alpha=0.001 11m
# err_train: 3434.437815640683 err_val: 5599.2871695679
# MLP (160,80,70,50) tol=0.00005 alpha=0.001 36m
# err_train: 3266.782885785861 err_val: 5571.2599459496205

# FULL TRAIN
# MLP (100,75,50), alpha=0.001 15m
# err_train: 3790.3653802664085 PUBLIC 7800.42

model = MLPRegressor(
    hidden_layer_sizes=(100, 75, 50),
    # tol=0.00005,
    alpha=0.001,
    verbose=True,
    random_state=42,
)
# %%
# On train le modèle
model.fit(X_train, y_train)
os.system('say "Wake up wake up, training complete!"')


# %%
# On récupère les prédictions sur train/test et on les formate en pandas
def get_df_result(dff, X_to_test):
    df_temp = pd.merge(
        dff.reset_index(),
        pd.DataFrame(model.predict(X_to_test), columns=["Load_pred"]),
        left_index=True,
        right_index=True,
    )
    rescale(df_temp, "Load_pred")
    rescale(df_temp, "Load")
    return df_temp


df_train_result = get_df_result(df_train, X_train)
df_test_result = get_df_result(df_test, X_test)


# %%
# Calcul de l'erreur train +- test si FULL_TRAIN ou non
def err(dff, col_true, col_pred):
    sum = 0
    for zone in li_zones:
        res = root_mean_squared_error(
            dff.loc[dff[zone] == 1, col_true], dff.loc[dff[zone] == 1, col_pred]
        )
        sum += res
    return sum


err_train = err(df_train_result, "Load", "Load_pred")
if FULL_TRAIN:
    print("err_train:", err_train)
else:
    err_test = err(df_test_result, "Load", "Load_pred")
    print("err_train:", err_train, "err_test", err_test)


# %%
# Plot du train et du test
def plot_pred(dff, zone, load=True):
    if load:
        plt.plot(dff.loc[dff[zone] == 1, "Load"], linewidth=1, alpha=0.8)
    plt.plot(
        dff.loc[dff[zone] == 1, "Load_pred"],
        linewidth=1,
        alpha=0.8,
    )
    plt.title(zone)
    plt.show()


zone_to_plot = "zone_France"
if FULL_TRAIN:
    plot_pred(df_train_result, zone_to_plot)
    plot_pred(df_test_result, zone_to_plot, load=False)
else:
    plot_pred(df_train_result, zone_to_plot)
    plot_pred(df_test_result, zone_to_plot)
# %%
# Output des prédictions vers un csv du format demandé
if FULL_TRAIN:
    to_keep = li_zones.copy()
    to_keep.append("Load_pred")
    to_keep.append("date")
    df_temp = df_test_result[to_keep]

    rename_mapping = {
        "zone_Montpellier": "pred_Montpellier Méditerranée Métropole",
        "zone_Lille": "pred_Métropole Européenne de Lille",
        "zone_Grenoble": "pred_Métropole Grenoble-Alpes-Métropole",
        "zone_Nice": "pred_Métropole Nice Côte d'Azur",
        "zone_Rennes": "pred_Métropole Rennes Métropole",
        "zone_Rouen": "pred_Métropole Rouen Normandie",
        "zone_Marseille": "pred_Métropole d'Aix-Marseille-Provence",
        "zone_Lyon": "pred_Métropole de Lyon",
        "zone_Nancy": "pred_Métropole du Grand Nancy",
        "zone_Paris": "pred_Métropole du Grand Paris",
        "zone_Nantes": "pred_Nantes Métropole",
        "zone_Toulouse": "pred_Toulouse Métropole",
    }

    for zone in li_zones:
        if zone in rename_mapping:
            new_col = rename_mapping[zone]
        else:
            new_col = "pred_" + zone.split("_", 1)[1]
        df_temp[new_col] = df_temp["Load_pred"] * df_temp[zone]

    desired_order = [
        "pred_France",
        "pred_Auvergne-Rhône-Alpes",
        "pred_Bourgogne-Franche-Comté",
        "pred_Bretagne",
        "pred_Centre-Val de Loire",
        "pred_Grand Est",
        "pred_Hauts-de-France",
        "pred_Normandie",
        "pred_Nouvelle-Aquitaine",
        "pred_Occitanie",
        "pred_Pays de la Loire",
        "pred_Provence-Alpes-Côte d'Azur",
        "pred_Île-de-France",
        "pred_Montpellier Méditerranée Métropole",
        "pred_Métropole Européenne de Lille",
        "pred_Métropole Grenoble-Alpes-Métropole",
        "pred_Métropole Nice Côte d'Azur",
        "pred_Métropole Rennes Métropole",
        "pred_Métropole Rouen Normandie",
        "pred_Métropole d'Aix-Marseille-Provence",
        "pred_Métropole de Lyon",
        "pred_Métropole du Grand Nancy",
        "pred_Métropole du Grand Paris",
        "pred_Nantes Métropole",
        "pred_Toulouse Métropole",
    ]

    result = (
        df_temp.groupby("date")[desired_order].sum().reset_index().set_index("date")
    )
    result.to_csv("data/pred.csv")

# %%
import joblib

joblib.dump(model, "models/MLP(100,75,50),alpha=0_001).joblib")
# %%
