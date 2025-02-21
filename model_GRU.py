# %%
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# %%

# Sample data in long format: each row is a time point for a given zone.
data = {
    "date": [
        "2022-01-01 00:00",
        "2022-01-01 00:30",
        "2022-01-01 00:00",
        "2022-01-01 00:30",
    ],
    "temp": [5, 6, 4, 3],
    "consumption": [100, 110, 90, 95],
    "zone": ["A", "A", "B", "B"],  # Zone identifier
}

# Create a DataFrame
df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
# %%

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
