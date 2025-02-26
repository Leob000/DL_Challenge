# %%
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader  # noqa: F401
import matplotlib.pyplot as plt

# %%
plt.rcParams["figure.figsize"] = [10, 5]

COLAB = False  # Si utilisation de google colab
FULL_TRAIN = True  # True: pred sur 2022, False: pred sur 2021 (validation)
STANDARDIZATION_PER_ZONE = True
DROP_AUGUSTS_FLAGS = True

if COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    df = pd.read_parquet("drive/MyDrive/data/clean_data.parquet")
else:
    df = pd.read_parquet("data/clean_data.parquet")
# %%
li_features = df.columns.to_list()
li_features.remove("zone")
li_features.remove("Load")
li_features


def get_zone_data(dataframe, target_col="Load"):
    data_dict = {}
    target_dict = {}
    grouped = dataframe.groupby("zone")
    for zone, group in grouped:
        group_sorted = group.sort_index()
        X = group_sorted[li_features].values
        # For training/validation, y is available.
        # For test, y might be NaN, so we'll only store X.
        y = (
            group_sorted[target_col].values
            if group_sorted[target_col].notna().all()
            else None
        )

        data_dict[zone] = torch.tensor(X, dtype=torch.float)
        if y is not None:
            target_dict[zone] = torch.tensor(y, dtype=torch.float)
    return data_dict, target_dict


if FULL_TRAIN:
    df_train = df[df.index < "2022"]
    df_test = df[df.index >= "2022"]
else:
    df_train = df[df.index < "2021"]
    df_test = df[(df.index >= "2021") & (df.index < "2022")]
    df_test.loc[:, "Load"] = None

X_train, y_train = get_zone_data(df_train, target_col="Load")
X_test, _ = get_zone_data(
    df_test, target_col="Load"
)  # test set: consumption is unknown

# Example: printing the shapes
for zone in X_train:
    print(f"Train Zone {zone} sequence X shape: {X_train[zone].shape}")
    print(f"Train Zone {zone} sequence y shape: {y_train[zone].shape}")

for zone in X_test:
    print(f"Test Zone {zone} sequence X shape: {X_test[zone].shape}")

# %%
#################### GPT example
df["date"] = pd.to_datetime(df["date"])

# Group by zone and sort each group by date
grouped = df.groupby("zone")
sequences = {}
for zone, group in grouped:
    group_sorted = group.sort_values("date")
    # Select features (e.g., temp and consumption); you can add more if needed.
    features = group_sorted[["temp", "consumption"]].values
    # Convert to tensor: shape (sequence_length, feature_dim)
    sequences[zone] = torch.tensor(features, dtype=torch.float)

sequences
# %%
# Print the sequences for verification
for zone, seq in sequences.items():
    print(f"Zone {zone} sequence shape: {seq.shape}")
    print(seq)


# %%
# Define a simple GRU encoder
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: shape (batch, sequence_length, input_dim)
        output, h = self.gru(x)
        return output, h


# Example: using the GRU encoder for each zone independently
input_dim = 2  # Number of features: temp and consumption
hidden_dim = 16  # Size of the GRU hidden state
encoder = GRUEncoder(input_dim, hidden_dim)

# %%
# Process each zone's sequence individually
for zone, seq in sequences.items():
    # Add a batch dimension: shape becomes (1, sequence_length, input_dim)
    seq_batch = seq.unsqueeze(0)
    output, hidden = encoder(seq_batch)
    print(f"Zone {zone}:")
    print("  Output shape:", output.shape)  # (1, seq_length, hidden_dim)
    print("  Hidden shape:", hidden.shape)  # (num_layers, 1, hidden_dim)

# %%
# TODO padding
# TODO Dataloader?
