Treatment:
- Débruitage Nancy/Nice au lieu d'enlever? Pour Nancy, possible d'implémanter dans la loss un seuil pour moins prendre en compte les données bruitées/non bruitées?
- Sinon possible de faire un modèle descendant, France -> Régions -> Villes, et donc train les villes à la fin juste à partir de la période où les données sont présentes
- Enlever les valeurs abérantes villes
- Voir le cas de Nancy, corrélation cheloue
- Date format en UTC? format FR?
- Voir comment traiter les NaN -> Faire un modèle pour deviner les NaN des villes, avec notamment les données météo?

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