# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["figure.figsize"] = [10, 5]

COLLAB = False
FULL_TRAIN = False
if COLLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    df = pd.read_parquet("drive/MyDrive/data/clean_data.parquet")
else:
    df = pd.read_parquet("data/clean_data.parquet")
# %%
# On normalise les variables continues, on choisit pour l'instant de normaliser par région
# On garde en liste les moyennes et std des Loads des différentes régions
li_zones = [
    "zone_Auvergne-Rhône-Alpes",
    "zone_Bourgogne-Franche-Comté",
    "zone_Bretagne",
    "zone_Centre-Val de Loire",
    "zone_France",
    "zone_Grand Est",
    "zone_Grenoble",
    "zone_Hauts-de-France",
    "zone_Lille",
    "zone_Lyon",
    "zone_Marseille",
    "zone_Montpellier",
    "zone_Nancy",
    "zone_Nantes",
    "zone_Nice",
    "zone_Normandie",
    "zone_Nouvelle-Aquitaine",
    "zone_Occitanie",
    "zone_Paris",
    "zone_Pays de la Loire",
    "zone_Provence-Alpes-Côte d'Azur",
    "zone_Rennes",
    "zone_Rouen",
    "zone_Toulouse",
    "zone_Île-de-France",
]
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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

# Seed 42 sauf si précisé
# MLP 50
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

model = MLPRegressor(
    hidden_layer_sizes=(100, 75, 50),
    # tol=0.00005,
    alpha=0.001,
    verbose=True,
    random_state=42,
)
# %%
model.fit(X_train, y_train)


# %%
def get_df_result(dff, X_to_test):
    df_temp = pd.merge(
        dff.reset_index(),
        pd.DataFrame(model.predict(X_to_test), columns=["Load_pred"]),
        left_index=True,
        right_index=True,
    )
    rescale(df_temp, "Load_pred")
    if not FULL_TRAIN:
        rescale(df_temp, "Load")
    return df_temp


df_train_result = get_df_result(df_train, X_train)
df_test_result = get_df_result(df_test, X_test)


# %%
# TODO setup truc automatique pour sortir format pour upload sur le site
# df_test_result.set_index("date").drop(
#     columns=["Load", "ff", "tc", "u", "is_ville", "is_pays", "is_reg", "temp_below"]
# )


# %%
df_train_result.loc[df_train_result["zone_France"] == 1, "Load"]
df_train_result.loc[df_train_result["zone_France"] == 1, "Load_pred"]
for zone in li_zones:
    a = df_train_result.loc[df_train_result[zone] == 1, "Load"]
    print(zone, a.shape)


# %%
def err(dff, col_true, col_pred):
    sum = 0
    for zone in li_zones:
        res = root_mean_squared_error(
            dff.loc[dff[zone] == 1, col_true], dff.loc[dff[zone] == 1, col_pred]
        )
        sum += res
    return sum


err_train = err(df_train_result, "Load", "Load_pred")
err_val = err(df_test_result, "Load", "Load_pred")
print("err_train:", err_train, "err_val:", err_val)


# %%
def plot_pred(dff, zone):
    plt.plot(dff.loc[dff[zone] == 1, "Load"], linewidth=1, alpha=0.8)
    plt.plot(
        dff.loc[dff[zone] == 1, "Load_pred"],
        linewidth=1,
        alpha=0.8,
    )
    plt.title(zone)
    plt.show()


# Train and valplot
for i in [df_train_result, df_test_result]:
    plot_pred(i, "zone_Grenoble")

# %%
