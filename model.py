# %%
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

np.random.seed(42)

# %%
df = pd.read_parquet("data/geo_tweaked.parquet")

# %%
X = df[["dayofweek", "dayofyear"]].values

tscv = TimeSeriesSplit()
model = RandomForestRegressor(random_state=42)

zone_list = ["France"]
for zone in zone_list:
    y = df[zone].values
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        err_train = root_mean_squared_error(y_train, model.predict(X_train))
        err_test = root_mean_squared_error(y_test, y_pred)
        print(f"{zone}, fold {fold}: err_train={err_train:.3f} err_test={err_test:.3f}")
        fold += 1
        if zone == "France":
            plt.plot(y_test, linewidth=1, alpha=0.5)
            plt.plot(y_pred, linewidth=1, alpha=0.8)
            plt.show()
