# Installation

## 1. Dataiku DSS

Installer Dataiku DSS Free Edition (launcher Windows ou distribution Linux/macOS) depuis https://www.dataiku.com/product/get-started.

## 2. Données Home Credit

1. Créer un compte Kaggle et accepter les règles de la compétition *Home Credit, Credit Risk Model Stability* (onglet Data).
2. Générer un token API (Kaggle > Settings > Create New Token) et placer `kaggle.json` dans `%USERPROFILE%\.kaggle\`.
3. Lancer `scripts\download_data.ps1`.

Seul le dossier `parquet_files/` et `feature_definitions.csv` sont conservés dans `data/raw/homecredit/`.

## 3. Réclamations CFPB

Télécharger depuis la page de l'archive des narratives les exports 1, 4, 20 et 21, puis les extraire chacun dans `data/raw/cfpb/<nom de l'export>/`.

## 4. Contrôle

```
python scripts/check_data.py
```

## 5. Connexion Dataiku

Administration > Connections > New connection > Filesystem, nom `creditops_data`, root path = dossier `data` du dépôt. Puis importer le projet depuis `dataiku/`.
