# %%
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim

FULL_TRAIN = False
DROP_AUGUSTS_FLAGS = True
DROP_PRECIPITATIONS = True
DROP_PRESSION = True
# %%
df = pd.read_parquet("data/clean_data.parquet")
if torch.backends.mps.is_available():
    device = torch.device("mps")  # MacOS GPU
print(device)
# %%
df.columns
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
df_train_france = df_train.loc[(df_train["zone"] == "France")]
df_test_france = df_test.loc[(df_test["zone"] == "France")]


# %%
# 1. Define the Dataset for training
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length, feature_cols, target_col):
        """
        df: pandas DataFrame with date-indexed data.
        seq_length: number of past time-steps used for each prediction.
        feature_cols: list of columns used as inputs. The first column should be "Load".
        target_col: name of the target column ("Load").
        """
        self.data = df[feature_cols].to_numpy(
            dtype="float32"
        )  # shape: (n_samples, n_features)
        # Here we assume "Load" is available in the features during training.
        self.targets = df[target_col].to_numpy(dtype="float32").reshape(-1, 1)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # x: sequence of shape (seq_length, n_features)
        x = self.data[idx : idx + self.seq_length]
        # y: next time-step "Load" value (target)
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# Suppose df is your preprocessed DataFrame (indexed by date) with columns including "Load" and exogenous features.
# For illustration, assume:
feature_cols = [
    "Load",
    "ff",
    "tc",
    "u",
    "temp_below_15",
    "tc_ewm15",
    "tc_ewm06",
    "tc_ewm15_max24h",
    "tc_ewm15_min24h",
    "sin_time",
    "cos_time",
    "sin_dayofweek",
    "cos_dayofweek",
    "sin_dayofyear",
    "cos_dayofyear",
    "is_weekend",
    "winter_hour",
    "is_holiday",
]
target_col = "Load"
seq_length = 48  # e.g., use past 24 hours (48 time-steps at 30min intervals)


# Create the training dataset and dataloader
train_dataset = TimeSeriesDataset(df_train_france, seq_length, feature_cols, target_col)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# %%


# 2. Define the LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.05):
        """
        input_size: number of input features (e.g., 1 (Load) + exogenous features).
        hidden_size: number of LSTM units.
        num_layers: number of stacked LSTM layers.
        dropout: dropout probability between layers.
        """
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)  # output a single value (Load)

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_length, input_size)
        lstm_out, hidden = self.lstm(x, hidden)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)  # shape: (batch, 1)
        return pred, hidden


# 3. Training loop with RMSE loss
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred, _ = model(x)
        # RMSE loss: sqrt(MSE)
        loss = nn.MSELoss()(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # TEST
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


# 4. Iterative forecasting (autoregressive) function
def iterative_forecast(model, init_seq, future_exog, device):
    """
    model: trained LSTMForecaster.
    init_seq: tensor of shape (seq_length, input_size) containing the last observed values.
              Here, the first element in each vector is "Load".
    future_exog: numpy array of shape (forecast_horizon, n_exog) that contains the future exogenous features.
                 These features should align with the order used during training (i.e. all features except "Load").
    device: torch device (e.g., "cpu" or "cuda").

    Returns a list of predicted "Load" values for the entire forecast horizon.
    """
    model.eval()
    predictions = []
    # We'll use a sliding window approach.
    seq = init_seq.clone().to(device)  # shape: (seq_length, input_size)
    # The number of features in the input vector:
    input_size = seq.shape[1]

    # Assume that in the input vector, index 0 is "Load" and indices 1: are the exogenous features.
    for t in range(len(future_exog)):
        # Prepare input: add batch dimension
        x_input = seq.unsqueeze(0)  # shape: (1, seq_length, input_size)
        with torch.no_grad():
            pred, _ = model(x_input)
        pred_value = pred.squeeze().item()
        predictions.append(pred_value)

        # Build new input for the next time step:
        # Copy the last vector from the current sequence and update it.
        new_entry = seq[-1].clone()
        new_entry[0] = pred_value  # update "Load" with the predicted value
        # Replace the exogenous features with the known future values at time t.
        # Make sure the order of features in future_exog matches indices 1: of the input vector.
        new_entry[1:] = torch.tensor(future_exog[t], dtype=torch.float32).to(device)

        # Update the sliding window: remove the oldest time-step and append the new one.
        seq = torch.cat([seq[1:], new_entry.unsqueeze(0)], dim=0)
    return predictions


# %%


input_size = len(feature_cols)
hidden_size = 64
num_layers = 7

model = LSTMForecaster(input_size, hidden_size, num_layers, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop (simplified)
epochs = 6
for epoch in range(epochs):
    loss = train_model(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}/{epochs}, RMSE: {loss:.4f}")

# %%

# Forecasting:
# Prepare the last observed window from your training or validation set.
# For example, take the last seq_length entries of df_train_france.
last_seq = torch.tensor(
    df_train_france[feature_cols].to_numpy(dtype="float32")[-seq_length:],
    dtype=torch.float32,
)
# Prepare future exogenous features for the forecast horizon.
# They should have shape (forecast_horizon, n_exog) where n_exog = input_size - 1.
# Here, you already know the values for 2022.
future_exog = df_test_france[
    [
        "ff",
        "tc",
        "u",
        "temp_below_15",
        "tc_ewm15",
        "tc_ewm06",
        "tc_ewm15_max24h",
        "tc_ewm15_min24h",
        "sin_time",
        "cos_time",
        "sin_dayofweek",
        "cos_dayofweek",
        "sin_dayofyear",
        "cos_dayofyear",
        "is_weekend",
        "winter_hour",
        "is_holiday",
    ]
].to_numpy(dtype="float32")

predictions = iterative_forecast(model, last_seq, future_exog, device)
print("Forecasts for the horizon:", predictions[:10], "...")

# %%
tuple_france = li_load[0]
mean_france, std_france = tuple_france[1], tuple_france[2]

predictions_unscale = []
for pred in predictions:
    pred = pred * std_france + mean_france
    predictions_unscale.append(pred)

print(predictions_unscale)
# %%
true = (
    df.loc[
        (df["zone"] == "France") & (df.index >= "2021") & (df.index < "2022"), "Load"
    ]
    * std_france
    + mean_france
)
true = true.to_numpy()
# %%
my_pred = np.array(predictions_unscale)
# %%
from sklearn.metrics import root_mean_squared_error

root_mean_squared_error(true, my_pred)
# %%
import matplotlib.pyplot as plt

plt.plot(true)
plt.plot(my_pred)
plt.show()

# %%
