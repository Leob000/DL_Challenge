# Todo:
- 🔥 Cleanup .md files, and TODO flags
- Handson
- Set up 2021 set for val set? less?

- Model architecture
    - 🔥 Loss adapted for NaN values?
    - Weighted date loss (Nancy), loss with lower weight for older data?
    - Find global model architecture, check SOTA model time series, FiLM for regions?, RNN?, ARIMA, ARMA, Prophet; Ensemble methods with non DL?
    - Model for France, then regions, then stations?
- Join files
    - Join method?
    - How to go from 3 hours to 30 min?
    - 🔥 Treat meteo NaN -> Interpolate by dept?
- Feature engineering:
    - 🔥 Check other variables from Goude project, especially variables from EDF formula
    - Date
        - How to categorize date, year/timeofyear, month/day of month (for august), weekday, hour/min...
        - Fourrier stuff? how to best extract seasonality? Prophet can do or other?
        - Lagged year -1, ..., -n values?
        - No lagged day value because of whole year? Estimate other model then takes its output as input for NN?
        - Holidays, single holiday
    - Meteo
        - Trend, moving average?
    - Feature scaling / categorizing check
- Do not forget:
    - Go back to timezone +1 and +2 before uploading pred file
    - Nice reshifted for last months, check with public score if we have to unshift the preds or not
    - National soberty plan, announced 2022-06, in action from 2022-10; underestimate the model from a certain date to get better results?

# Project summary:
- Data:
    - Electricity consumption from ~2017 to the end of 2021 (data point every 30 min), for France, its 12 regions, and 12 metropoles.
    - Various meteo data from 40 meteo stations, from ~2017 to 2022-12-18 (data point every 3 hours). Every city metropole has a station close by, except for Grenoble (equidistant from 3 stations). Useful features at first glance: temperature, humidity, wind speed.
- Goal:
    - Predict electricity consumption for each zone (France, regions, metropoles) for the whole 2022 year (every 30 min).
    - Loss: RMSE