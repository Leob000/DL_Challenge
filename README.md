Pour ce Data challenge, nous disposons des informations suivantes :

- Données :
    - Consommation d'électricité en MegaWatt des ces différentes zones de 2017 (début des données variable pour les métropoles) à 2021 inclus.
    - Données météos variées d'une quarantaine de stations météo, de 2017 à 2022 inclus (avec intervalle de temps 3 heures).
- Objectif :
    - Prédire la consommation d'électricité pour l'année entière 2022, avec intervalle de temps 30 minutes, pour les différentes zones (la France, ses 12 régions, 12 de ses métropoles).
    - Le critère d'évaluation de la prédiction se fera grâce à la somme des RMSE pour chaque région. Nous notons donc que plus les valeurs de consommation moyenne d'une zone sera importante, plus celle-ci aura un poids important dans le critère d'évaluation.

Nous organisons notre code de la manière suivante (il faut executer ces fichiers dans l'ordre pour reproduire nos résultats) :

- Tout d'abord deux fichiers `explo_geo.py` et `explo_meteo.py` qui output les données traitées, respectivement de la consommation électrique et de la météo.
- Vient ensuite `feature_eng.py`, qui output la jointure des deux jeux de données sur laquelle a été fait du feature engineering.
- Le fichier `model_MLP.py` où nous effectuons quelques modifications finales des données, puis soit nous chargeons un modèle déjà créé, soit nous créons un nouveau modèle et, si celui-ci est un modèle entraîné sur toutes les données (donc pas juste pour validation), on le sauvegarde. Pour les modèles entraîné sur toutes les données, on créé un fichier csv des prédictions dans le format attendu par le challenge. Nous avons aussi implémenté un LSTM dans le fichier `model_LSTM.py`.

La version de python utilisée est la 3.12. Les packages requis sont trouvés dans `requirements.txt`.

Attention, les données du challenge doivent être dans un dossier `data/`, les modèles sont sauvegardés et chargés à partir du dossier `models/`.

Différentes options (affichage des graphiques, utilisation de colab, ...) sont possibles pour certains fichiers, et mises sous la forme de constantes booléennes au début de ceux-ci. Un fichier `utils.py` est utilisé pour le stockage de fonctions d'affichage géographique lors de l'exploration des données.