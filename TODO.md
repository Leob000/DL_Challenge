Feature engineering:
- Lagged values 1, 7 -> Ou onehotencoding days of week, months, ... check avec gpt vs lagged values, fourrier stuff...
    - Model dayofweek with onehotencoding, dayofyear with fourrier?, hourly?
    - Deseasonalize data before lagged values?
- Weekends, Friday?
- Daylight savings
- Holidays, Férié, Aout
- Trend, Moving average

Global:
- Réfléchir quel modèle faire, quoi et comment modéliser
- FiLM modèle?
- Voir avec chatGPT pour des architectures qui pourraient marcher, RNN marchent bien avec time series?
- Ajouter les données météo petit à petit, pas d'un coup?

- ARIMA
- ARMA
- Ensemble methods to correct DL ?