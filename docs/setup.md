# Installation

## 1. Dataiku DSS

Installer Dataiku DSS Free Edition (launcher Windows ou distribution Linux/macOS) depuis https://www.dataiku.com/product/get-started.

## 2. Données

| Source | Fichiers à placer |
| --- | --- |
| [Kaggle, Home Credit CRMS](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) (règles de la compétition à accepter) | `parquet_files/` et `feature_definitions.csv` dans `data/raw/homecredit/` |
| [CFPB, archive des narratives](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) | Exports 1, 4, 20 et 21, chacun extrait dans `data/raw/cfpb/<nom de l'export>/` |

## 4. Contrôle

```
python scripts/check_data.py
```

## 5. Projet Dataiku

Créer une clé d'API personnelle (Profile & Settings > API keys), l'enregistrer dans `~/.creditops_dss_key`, puis lancer avec le Python fourni par DSS :

```
python scripts/setup_dss_project.py --host http://localhost:11200
```

Le script crée la connexion `creditops_data`, le projet, ses zones, ses variables et les datasets sources. Il peut être relancé sans risque.

Alternative : importer directement `dataiku/CREDITOPS360.zip` (Import project depuis la page d'accueil DSS). L'archive contient la configuration complète (Flow, recettes, modèle `pd_model` v2, Knowledge Bank, agents, scénario) mais aucune donnée : les datasets sont à reconstruire après avoir placé les fichiers sources dans `data/raw/`. Les échantillons d'entraînement du modèle sont retirés de l'archive, l'onglet d'explication par lignes du modèle sauvegardé n'est donc disponible qu'après réentraînement.
