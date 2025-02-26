# %%
import torch
import pandas as pd
from torch.utils.data import Dataset

COLAB = False
FULL_TRAIN = False
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
