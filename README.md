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
| 2. Qualité | Recette de validation bloquante (`hc_base_validated`), règles Data Quality, métrique Python personnalisée (Gini, PSI) |
| 3. Features | Agrégation Python des tables lourdes (210 M de lignes, lecture par fichier), recettes Group visuelles sur les tables légères, jointure sur `case_id` |
| 4. ML | Statistiques interactives, Visual ML, explicabilité, analyse d'équité, scoring |
| 5. Monitoring | Gini, PSI et score de stabilité par semaine, Evaluate recipe vers un Model Evaluation Store (dérive des données et des prédictions), service API `credit_scoring` |
| 6. Réclamations | Classification, résumé et extraction d'entités par LLM, évaluation contre les labels CFPB |
| 7. Réglementaire | Knowledge Bank (EBA, AI Act, ACPR), RAG |
| 8. Agents | Outils (recherche réglementaire, profil de dossier, statistiques de réclamations), agent `copilote_analyste`, agent structuré `revue_dossier` |

Automatisation : le scénario `weekly_retrain` valide les données, recalcule le monitoring et ne réentraîne le modèle qu'en cas d'alerte (PSI au-delà de 0,2 ou Gini en baisse de plus de 5 points). Le service API `credit_scoring` est conçu et testé sur le nœud Design ; son déploiement (bundle, nœuds Automation et API) est décrit dans [`docs/mlops.md`](docs/mlops.md).

## Résultats

| Brique | Résultat |
| --- | --- |
| Données | 1 526 659 dossiers, 17 tables, 210 M de lignes agrégées en 496 variables |
| Modèle de défaut v2 (LightGBM calibré, test temporel) | AUC 0,824, Gini 0,65, log loss 0,124 (0,457 en v1) ; seuil 0,125 issu d'une matrice de coûts |
| Stabilité en production simulée (semaines 41 à 91) | Gini moyen 0,63, score de stabilité 0,61, PSI du score max 0,049 |
| Model Evaluation Store (semaines 80 à 91) | AUC 0,81, log loss 0,126 ; v2 entraînée sans `date_decision` après détection de la dérive en v1 |
| Triage des réclamations par LLM (206 cas) | JSON valide 100 %, bon produit 62 % (prompt libre) puis 72 % (classification contrainte few-shot) |
| RAG réglementaire | 1 274 passages (EBA, AI Act, directive crédit conso, ACPR), réponses citées avec document et page |
| Agents | Copilote analyste et revue de dossier en français, réponses vérifiées sur 4 questions de test |

L'ensemble de la partie GenAI tourne **en local** : modèles servis par Ollama (qwen2.5 7B, bge-m3) et branchés au LLM Mesh par un plugin de connexion maison (`dataiku/plugins/ollama-llm`).

## Documentation

- [`docs/setup.md`](docs/setup.md) : reconstituer l'environnement
- [`docs/agents.md`](docs/agents.md) : outils, agents et tests
- [`docs/mlops.md`](docs/mlops.md) : évaluation, service API, déploiement

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
dataiku/python/     bibliothèque projet (agrégations, métriques de stabilité)
dataiku/recipes/    sources des recettes Python
dataiku/plugins/    plugin de connexion LLM Mesh vers Ollama
dataiku/metrics/    métrique personnalisée
dataiku/scenarios/  étape Python du scénario de réentraînement
scripts/            mise en place du projet DSS par l'API (idempotente) et contrôle des données
docs/               installation, choix techniques
data/               données locales, ignorées par Git
```

## Stack

Dataiku DSS 15 (Visual ML, recettes visuelles et Python, Data Quality, scénarios, Model Evaluation Store, API Designer, LLM Mesh, Prompt Studio, Knowledge Bank, agents, plugins, code envs), Python (pandas, pyarrow, pypdf), Ollama.
