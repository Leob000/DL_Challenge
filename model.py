# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
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
df_train = df[df.index < "2021"]
df_val = df[(df.index >= "2021") & (df.index < "2022")]
df_test = df[df.index >= "2022"]

X_train = df_train.drop(columns=["Load"]).to_numpy(dtype="float32")
y_train = df_train["Load"].to_numpy(dtype="float32")

X_val = df_val.drop(columns=["Load"]).to_numpy(dtype="float32")
y_val = df_val["Load"].to_numpy(dtype="float32")

X_train = df_train.drop(columns=["Load"]).to_numpy(dtype="float32")
y_train = df_train["Load"].to_numpy(dtype="float32")

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error

model = MLPRegressor(hidden_layer_sizes=(100, 75, 50), verbose=True, random_state=42)
# %%
model.fit(X_train, y_train)

# %%
df_val_result = pd.merge(
    df_val.reset_index(),
    pd.DataFrame(model.predict(X_val), columns=["Load_pred"]),
    left_index=True,
    right_index=True,
)
rescale(df_val_result, "Load_pred")
rescale(df_val_result, "Load")

df_train_result = pd.merge(
    df_train.reset_index(),
    pd.DataFrame(model.predict(X_train), columns=["Load_pred"]),
    left_index=True,
    right_index=True,
)
rescale(df_train_result, "Load_pred")
rescale(df_train_result, "Load")


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
err_val = err(df_val_result, "Load", "Load_pred")
print("err_train:", err_train, "err_val:", err_val)
# %%
plt.rcParams["figure.figsize"] = [10, 5]
plt.plot(
    df_val_result.loc[df_val_result["zone_France"] == 1, "Load"], linewidth=1, alpha=0.8
)
plt.plot(
    df_val_result.loc[df_val_result["zone_France"] == 1, "Load_pred"],
    linewidth=1,
    alpha=0.8,
)
plt.show()

plt.plot(
    df_train_result.loc[df_train_result["zone_France"] == 1, "Load"],
    linewidth=1,
    alpha=0.8,
)
plt.plot(
    df_train_result.loc[df_train_result["zone_France"] == 1, "Load_pred"],
    linewidth=1,
    alpha=0.8,
)
plt.show()
# %%
model.n_layers_
# (100,) err_train: 4970.189566458095 err_val: 5305.83536908326
