# Compétition Kaggle - IFT 3395

Ce document vous guide à travers l'exécution du code pour la compétition Kaggle IFT 3395.

## Exécution du Code

### Régression Logistique

Pour entraîner le modèle, calculer la précision dans l'ensemble de validation et générer le fichier 'predictions.csv' contenant les prédictions sur le fichier "test.csv", suivez simplement ces étapes :

1. Ouvrez un terminal.
2. Exécutez la commande suivante :

```sh
python3 regression_logistique/train.py
```

Le script `train.py` affiche également des informations utiles, telles que la distribution des classes prédites dans l'ensemble de validation, la précision dans l'ensemble de validation, et la distribution des classes dans l'ensemble d'entraînement.

Vous pouvez personnaliser certains paramètres du modèle en modifiant les attributs suivants :

- `n_iter`: Le nombre d'itérations pour l'apprentissage.
- `lr`: Le taux d'apprentissage.
- `balance`: Le pourcentage d'éléments prélevés de la classe majoritaire (classe 0).

Vous avez également la possibilité de changer le nom du fichier CSV de sortie. 

### Grid Search

Pour effectuer une recherche en grille (grid search) des paramètres optimaux, suivez ces étapes :

1. Ouvrez un terminal.
2. Exécutez la commande suivante :

```sh
python3 regression_logistique/grid_search.py
```

Vous pouvez personnaliser les éléments de recherche en modifiant les tableaux 'learning_rate', 'iterations' et 'balance_percents' sous 'if __name__ == "__main__"'. Les résultats de la recherche seront enregistrés dans le fichier "grid_search_results.csv".


### Classification Naive Bayes - Kaggle IFT 3395

Ce document vous guide à travers l'exécution du code pour la compétition Kaggle IFT 3395 en utilisant un modèle de Classification Naive Bayes.

## Exécution du Code

Entraînement du Modèle
Pour entraîner le modèle, effectuer des prédictions sur l'ensemble de test et générer le fichier 'GNB.csv' contenant les prédictions, veuillez suivre les étapes suivantes :

Ouvrez un terminal.
Exécutez la commande suivante :

```sh
python3 naive_bayes/naiveOO.ipynb
```

Le script `naiveOO.ipynb` implémente un classificateur Naive Bayes gaussien pour des données continues. Il calcule les statistiques nécessaires à partir des données d'entraînement, puis utilise ces statistiques pour prédire les classes des échantillons dans l'ensemble de test.

Les prédictions seront enregistrées dans le fichier "GNB.csv" dans le répertoire de sortie.

### Visualisation des Résultats
Le script génère également des visualisations pour évaluer les performances du modèle :

Une matrice de confusion pour évaluer les performances du modèle.
Un histogramme pour afficher la distribution des étiquettes prédites.
Des graphiques Q-Q pour vérifier la normalité des caractéristiques.
Les résultats et visualisations sont générés dans le terminal et affichés à l'utilisateur pour évaluer la performance du modèle.

Il est recommandé de personnaliser les paramètres du modèle, si nécessaire, dans le script train.py.

### Utilisation

Pour utiliser le modèle Naive Bayes entraîné sur d'autres données, vous pouvez suivre les étapes suivantes :

Importez le modèle et les fonctions nécessaires.
Chargez vos propres données dans un format similaire à celui des données d'entraînement.
Utilisez le modèle pour faire des prédictions sur vos données.
Vous pouvez également utiliser les fonctions de visualisation pour évaluer les résultats de vos prédictions.
N'oubliez pas de personnaliser le chemin du fichier de sortie pour enregistrer vos prédictions.

Ce modèle de Classification Naive Bayes gaussien peut être un outil puissant pour la classification de données continues. Il est prévu pour fonctionner avec des données similaires à celles de la compétition Kaggle IFT 3395, mais il peut être adapté pour d'autres ensembles de données en ajustant les paramètres et la préparation des données.