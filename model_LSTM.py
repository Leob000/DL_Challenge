# %%
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim

COLAB = False
UNE_ZONE = True  # On entraine juste pour la France

FULL_TRAIN = False

BATCH_SIZE = 128
SEQ_LENGTH = 48 * 2  # e.g., use past 24 hours (48 time-steps at 30min intervals)
HIDDEN_SIZE = 64 * 2  # LSTM
NUM_LAYERS = 7  # LSTM
EPOCHS = 8
CLIP_GRAD = False

feature_cols = [
    "Load",
    "tc",
    "ff",
    "u",
    "tc_ewm15",
    "tc_ewm06",
    "tc_ewm15_max24h",
    "tc_ewm15_min24h",
    "temp_below_15",
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
# %%
device = torch.device("cpu")
if COLAB:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    df = pd.read_parquet("drive/MyDrive/data/clean_data.parquet")
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Colab GPU
else:
    df = pd.read_parquet("data/clean_data.parquet")
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # MacOS GPU

print("Device used:", device)
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


# %%
# Séparation du dataset en train et test set
if FULL_TRAIN:
    df_train = df[df.index < "2022"]
    df_test = df[df.index >= "2022"]
else:
    DATE_THRESHOLD = "2021"
    df_train = df[df.index < "2021"]
    df_test = df[(df.index >= "2021") & (df.index < "2022")]
    df_train = df[df.index < DATE_THRESHOLD]
    df_test = df[(df.index >= DATE_THRESHOLD) & (df.index < "2022")]
    df_test.loc[:, "Load"] = float("nan")  # On retire les y du test de validation


# %%
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
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, idx):
        # x: sequence of shape (seq_length, n_features)
        x = self.data[idx : idx + self.seq_length]
        # y: next time-step "Load" value (target)
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
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
        if CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # TEST
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)


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
    # input_size = seq.shape[1]

    # Assume that in the input vector, index 0 is "Load" and indices 1: are the exogenous features.
    for t in range(len(future_exog)):
        # Prepare input: add batch dimension
        x_input = seq.unsqueeze(0).to(device)  # shape: (1, seq_length, input_size)
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


err_validation = 0
results = pd.DataFrame()
for zone, mean, std in li_load:
    print(zone)
    df_train = df_train.loc[(df_train["zone"] == zone)]
    df_test = df_test.loc[(df_test["zone"] == zone)]

    train_dataset = TimeSeriesDataset(df_train, SEQ_LENGTH, feature_cols, target_col)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_size = len(feature_cols)

    model = LSTMForecaster(input_size, HIDDEN_SIZE, NUM_LAYERS, dropout=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        loss = train_model(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS}, RMSE: {loss:.4f}")

    # Window des dernières données observées
    last_seq = torch.tensor(
        df_train[feature_cols].to_numpy(dtype="float32")[-SEQ_LENGTH:],
        dtype=torch.float32,
    )

    # Données exogènes pour les prédictions
    # They should have shape (forecast_horizon, n_exog) where n_exog = input_size - 1.
    feature_no_load = feature_cols.copy()
    feature_no_load.remove("Load")
    future_exog = df_test[feature_no_load].to_numpy(dtype="float32")

    predictions = iterative_forecast(model, last_seq, future_exog, device)

    predictions_unscale = []
    for pred in predictions:
        pred = pred * std + mean
        predictions_unscale.append(pred)
    my_pred = np.array(predictions_unscale)

    results[zone] = my_pred

    if not FULL_TRAIN:
        true = (
            df.loc[
                (df["zone"] == "France")
                & (df.index >= DATE_THRESHOLD)
                & (df.index < "2022"),
                "Load",
            ]
            * std
            + mean
        )
        true = true.to_numpy()
        err_validation_zone = root_mean_squared_error(true, my_pred)
        print("err_validation pour", zone, ":", err_validation_zone)
        err_validation += err_validation_zone
        plt.plot(true)
        plt.plot(my_pred)
        plt.show()
    if UNE_ZONE:
        break

if (not FULL_TRAIN) and (not UNE_ZONE):
    print("err_validation totale:", err_validation)
