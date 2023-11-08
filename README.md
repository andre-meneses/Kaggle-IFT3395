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

