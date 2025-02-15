# Explo
- Traiter l'index
- Traitrer les NaN faibles
- Date format UTC? autre?
- Faire checklist Handson
- Traitement des données météo
    - Liens stations-villes, stations-régions, stations-France
- Loss pour prendre en compte NaN ?

- Faire schéma global du modèle ?


Pré-modèle

# Modèle
- Nancy/Autres: implémanter dans la loss un poids pour prendre plus en compte les données récentes?
- Pour les prédictions, voir sur public score si meilleur en reshiftant ou non Nice
- Modèle NN descendant France -> Régions -> Villes, train les villes à partir du début de leur données


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