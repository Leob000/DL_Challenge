# Todo:
- Modèle initial prohet puis autres? possible de faire gros MLP sur toutes les données et toutes les sorties? MLP regression avec plusieurs output de regression ? Modèle fine-tune par région/autre pour éviter de retrain le fait d'apprendre les temporalités à chaque fois?
- Normalisation problem, France is the same weight as metropoles; make a weighted loss, weigth = mean electricity consumption?

- TFT
    - One hot encode region, station/city
    - Interpolate weather data
    - NaN 2017?

- Model architecture
    - (Loss adapted for NaN values?)
    - Weighted date loss (Nancy), loss with lower weight for older data?
    - Find global model architecture, check SOTA model time series, FiLM for regions?, RNN?, ARIMA, ARMA, Prophet; Ensemble methods with non DL?
    - Model for France, then regions, then stations?
- Join files
    - Join method?
- Feature engineering:
    - 🔥 Check other variables from Goude project, especially variables from EDF formula
    - Date
        - Categorize: Categorize month, day, hour? trigonometric circle? ask chatgpt
        - Trigonometry
        - Weekend flag
        - UTC, get DLS in hour data or just categorize
        - How to categorize date, year/timeofyear, month/day of month (for august), weekday, hour/min...
        - Fourrier stuff? how to best extract seasonality? Prophet can do or other?
        - Lagged year -1, ..., -n values?
        - No lagged day value because of whole year? Estimate other model then takes its output as input for NN?
        - Holidays, single holiday
    - Meteo
        - Ajouter precipitations, neige?
        <!-- - De base regrouper les régions en faisant moyenne classique, plus tard regrouper en faisant moyenne pondérée par pop des départements? -->
        <!-- - Onehotenc: Mountain, seaside, ... -->
        <!-- - Température ressentie, lissage exponentiel en feature engineering (temp passée, vent, humidité, ensolleillement) -->
    - Feature scaling / categorizing check, passer les colonnes en bool si nécessaire
    - Round les values?
- Do not forget:
    - Nice reshifted for last months, check with public score if we have to unshift the preds or not
    - National soberty plan, announced 2022-06, in action from 2022-10; underestimate the model from a certain date to get better results?
    - Standardization y and yhat, layernorm/batchnorm makes this useless?

# Project summary:
- Data:
    - Electricity consumption from ~2017 to the end of 2021 (data point every 30 min), for France, its 12 regions, and 12 metropoles.
    - Various meteo data from 40 meteo stations, from ~2017 to 2022-12-18 (data point every 3 hours). Every city metropole has a station close by, except for Grenoble (equidistant from 3 stations). Useful features at first glance: temperature, humidity, wind speed.
- Goal:
    - Predict electricity consumption for each zone (France, regions, metropoles) for the whole 2022 year (every 30 min).
    - Loss: RMSE
- Our features:
    - ...