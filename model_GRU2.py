# %%
import torch
import pandas as pd
from torch.utils.data import Dataset

COLAB = False
FULL_TRAIN = False
DROP_AUGUSTS_FLAGS = True
DROP_PRECIPITATIONS = True
DROP_PRESSION = True
# %%
# Chargement dataset/GPU (Colab/MacOS/CPU)

device = torch.device("cpu")  # CPU si pas de GPU dispo
if COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    df = pd.read_parquet("drive/MyDrive/data/clean_data.parquet")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Colab GPU
else:
    df = pd.read_parquet("data/clean_data.parquet")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # MacOS GPU

print("Device used:", device)
# %%
# Feature selection, on drop ou non certaines features
features_to_normalize = [
    "ff",
    "tc",
    "u",
    "tc_ewm15",
    "tc_ewm06",
    "tc_ewm15_max24h",
    "tc_ewm15_min24h",
]

df = df.drop(columns=["is_ville", "is_reg", "is_pays"])

if DROP_AUGUSTS_FLAGS:
    df = df.drop(columns=["is_august", "is_july_or_august"])

if DROP_PRECIPITATIONS:
    df = df.drop(columns=["rr1"])
else:
    features_to_normalize.append("rr1")

if DROP_PRESSION:
    df = df.drop(columns=["pres"])
else:
    features_to_normalize.append("pres")

# %%
# On normalise les variables continues
# On garde en liste les moyennes et std des Loads des différentes régions pour renormaliser les pred à la fin
li_zones = df["zone"].unique().tolist()

# Standardization per zone de "Load"
li_load = []
for zone in li_zones:
    load_temp = (
        zone,
        df.loc[df["zone"] == zone, "Load"].mean(),
        df.loc[df["zone"] == zone, "Load"].std(),
    )
    li_load.append(load_temp)
    df.loc[df["zone"] == zone, "Load"] = (
        df.loc[df["zone"] == zone, "Load"] - df.loc[df["zone"] == zone, "Load"].mean()
    ) / df.loc[df["zone"] == zone, "Load"].std()


# Standardization of the rest, per zone or globally
for zone in li_zones:
    for feature in features_to_normalize:
        df.loc[df["zone"] == zone, feature] = (
            df.loc[df["zone"] == zone, feature]
            - df.loc[df["zone"] == zone, feature].mean()
        ) / df.loc[df["zone"] == zone, feature].std()


# Fonction à utilier plus tard pour rescale
# TODO A changer plus tard en fonction de quand et comment rescale les trucs
def rescale(dff, col):
    for zone, mean, std in li_load:
        dff.loc[dff["zone"] == zone, col] = (
            dff.loc[dff["zone"] == zone, col] * std + mean
        )


# %%
# Séparation du dataset en train et test set
if FULL_TRAIN:
    df_train = df[df.index < "2022"]
    df_test = df[df.index >= "2022"]
else:
    df_train = df[df.index < "2021"]
    df_test = df[(df.index >= "2021") & (df.index < "2022")]
    df_test.loc[:, "Load"] = float("nan")  # On retire les y du test de validation


# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, df, zone):
        self.X = torch.tensor(
            df.loc[df["zone"] == zone]
            .drop(columns=["Load", "zone"])
            .to_numpy(dtype="float32")
        )
        self.y = torch.tensor(
            df.loc[df["zone"] == zone, "Load"].to_numpy(dtype="float32")
        )
        # self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # TODO Implémanter les transfos des valeurs (norm), les options de drop de variables

        obs = self.X[idx, :]
        label = self.y[idx]

        # if self.transform:
        #     obs = self.transform(obs)
        # if self.target_transform:
        #     label = self.target_transform(label)

        return obs, label


# %%
test_dataset = TimeSeriesDataset(df_test, "Paris")
# %%
test_dataset.__getitem__(0)
