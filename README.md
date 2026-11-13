# CreditOps 360

Plateforme de décision crédit pour une banque de détail, construite de bout en bout dans **Dataiku DSS** : scoring de défaut multi-tables, monitoring de dérive et réentraînement automatique, triage des réclamations clients par LLM, copilote réglementaire (RAG) et agents d'analyse de dossier.

## Scénario métier

La direction des risques veut industrialiser l'octroi de crédit à la consommation. Elle a besoin :

1. d'un score de probabilité de défaut **stable dans le temps**, exposé en API ;
2. d'un moyen de **justifier chaque décision** au regard des orientations EBA sur l'octroi de prêts et de l'AI Act (le scoring crédit y est classé à haut risque) ;
3. d'un outil pour **absorber le flux de réclamations** clients.

## Architecture du Flow

| Zone | Contenu |
| --- | --- |
| 1. Ingestion | Dossier des Parquet bruts, recette Python de conversion des tables légères en datasets (DSS ne lit pas le Parquet nativement sous Windows), dictionnaire de données |
| 2. Qualité | Data quality rules, métriques et checks |
| 3. Features | Agrégation Python des tables lourdes (210 M de lignes, lecture par fichier), recettes Group visuelles sur les tables légères, jointure sur `case_id` |
| 4. ML | Statistiques interactives, Visual ML, explicabilité, analyse d'équité, scoring |
| 5. Monitoring | Model Evaluation Store, dérive des données et des prédictions, PSI et Gini par semaine |
| 6. Réclamations | Classification, résumé et extraction d'entités par LLM, évaluation contre les labels CFPB |
| 7. Réglementaire | Knowledge Bank (EBA, AI Act, ACPR), RAG |
| 8. Agents | Copilote analyste, agent de revue de dossier, Agent Hub, webapp Credit Cockpit |

Mise en production : bundle publié via le Project Deployer, service API (prédiction + enrichissement par lookup) via l'API Deployer, réentraînement déclenché quand le PSI dépasse 0,2 ou que le Gini chute de plus de 5 points.

## Données

| Jeu | Usage | Licence |
| --- | --- | --- |
| [Home Credit, Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) | Scoring de défaut, dérive par semaine | Règles de la compétition Kaggle : non redistribué ici |
| [CFPB Consumer Complaint Database, archive des narratives](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/) | Triage des réclamations | Domaine public (CC0) |
| [EBA GL/2020/06](https://www.eba.europa.eu/regulation-and-policy/credit-risk/guidelines-on-loan-origination-and-monitoring), AI Act, publications ACPR | Corpus réglementaire | Réutilisation avec citation de la source |

Les données ne sont pas versionnées. Pour les reconstituer, voir [`docs/setup.md`](docs/setup.md).

## Structure du dépôt

```
dataiku/            export du projet DSS (configuration, recettes, code), sans données
dataiku/python/     code de la bibliothèque projet (agrégations, métriques de stabilité)
scripts/            téléchargement et contrôle des données
docs/               installation, choix techniques
data/               données locales, ignorées par Git
```

## Stack

Dataiku DSS (Visual ML, recettes visuelles et Python, scénarios, LLM Mesh, Knowledge Bank, agents, Project et API Deployer), Python (pandas, pyarrow).
