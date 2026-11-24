# MLOps : évaluation, API et déploiement

Objets créés par `scripts/setup_mlops.py` (idempotent) dans la zone **5 Monitoring** du projet `CREDITOPS360`.

## Architecture

```
hc_prod ──Filter (WEEK_NUM >= 80)──> hc_prod_recent ──Evaluate (pd_model)──> mes_pd_model (MES)
                                          │                              └──> mes_pd_model_metrics
                                          └──Sample (2 000 premiers)──> hc_enrich_cases ──> enrichissement API

Service API credit_scoring
  ├─ predict_default        : vecteur de features complet -> probabilité de défaut
  └─ predict_default_by_id  : {"case_id": ...} seul, features ajoutées par enrichissement (hc_enrich_cases, Bundled)
```

| Objet | Rôle |
| --- | --- |
| `compute_hc_prod_recent` | Filtre les 12 dernières semaines de production (165 k dossiers) pour limiter la mémoire de l'évaluation |
| `evaluate_pd_model` | Recette Evaluate : performance (le label `target` est connu), dérive des données et des prédictions par rapport au jeu d'entraînement |
| `mes_pd_model` | Model Evaluation Store : historique des évaluations, labels `scope` et `model` |
| `compute_hc_enrich_cases` | Table de lookup réduite (2 000 dossiers) embarquée dans le package API |
| `credit_scoring` v1 | Package API publié sur l'API Deployer local |

Commandes :

```
python scripts/setup_mlops.py                 # objets seuls
python scripts/setup_mlops.py --build-inputs --build   # (re)construit les entrées et lance l'évaluation
python scripts/setup_mlops.py --bundle        # bundle + package API, publication sur les deployers
```

## Résultats (évaluation du 24/11/2026, semaines 80 à 91)

| Métrique | Valeur |
| --- | --- |
| AUC | 0,811 (Gini 0,62) |
| Log loss | 0,450 |
| Précision / rappel (seuil 0,725) | 0,170 / 0,306 |
| Lift | 2,11 |
| Dérive des prédictions | PSI 0,0006, p-value chi² 0,98 : aucune dérive du score |
| Dérive des données (drift model) | accuracy 0,998 [0,998 ; 0,999], p-value 0 : dérive forte |

La dérive des données vient surtout de `date_decision` (utilisée comme feature d'entrée, PSI 6,2), de `refreshdate_3813885D_min` et des champs `maritalst` / `education` dont le codage change dans le temps (PSI 3,7). Le score reste stable et l'AUC se maintient : la dérive touche des variables peu importantes. Recommandation : retirer `date_decision` des features au prochain entraînement, sans quoi le drift model reste mécaniquement à 1.

### Modèle v2 (appliqué le 24/11/2026)

Suite à cette analyse, `pd_model` a été réentraîné puis redéployé dans le Flow (nouvelle version active) :

- `date_decision`, `age_band`, `case_id`, `MONTH` et `WEEK_NUM` en Reject (300 features d'entrée) ;
- calibration isotonique des probabilités ;
- LightGBM seul.

| Métrique | v1 | v2 |
| --- | --- | --- |
| AUC (test) | 0,821 | 0,824 |
| Log loss (test) | 0,457 | 0,124 |
| Seuil retenu (matrice de coûts) | 0,725 | 0,125 |
| AUC / log loss MES (semaines 80 à 91) | 0,811 / 0,450 | 0,810 / 0,126 |
| Gini moyen / stabilité / PSI max (semaines 41 à 91) | 0,633 / 0,616 / 0,045 | 0,630 / 0,613 / 0,049 |

Le pouvoir discriminant est inchangé ; le gain porte sur la calibration, les probabilités étant désormais interprétables comme des PD. Les tests API ci-dessous ont été faits sur la v1 : le package du service doit être régénéré pour servir la v2.

## Tests du service API (Design node, serveur de développement)

Requêtes de test sur `predict_default` avec trois dossiers réels de `hc_enrich_cases` :

| case_id | proba_1 API | proba_1 batch (`hc_prod_scored`) | Décision (seuil 0,725) | target |
| --- | --- | --- | --- | --- |
| 33494 | 0,575 | 0,569 | 0 | 0 |
| 33602 | 0,697 | 0,692 | 0 | 0 |
| 33615 | 0,660 | 0,646 | 0 | 1 |

Exemple de réponse :

```json
{"result": {"prediction": "0", "probaPercentile": 92, "probas": {"0": 0.3028, "1": 0.6972}}}
```

L'écart API / batch (moins de 0,015) vient du moteur de scoring Java de l'endpoint et de l'envoi de `date_decision` en chaîne ISO.

`predict_default_by_id` n'est pas testable localement : l'enrichissement « Bundled » charge la table dans une connexion SQL, et cette instance n'en a aucune (« No connection for bundled data in the dev server »). Tant que cet enrichissement est présent, le serveur de développement refuse de démarrer le service : les tests de `predict_default` ont été joués après son retrait temporaire, puis il a été remis.

## État du déploiement sur cette instance

| Élément | État |
| --- | --- |
| Project Deployer / API Deployer locaux | Présents (mode `LOCAL`), aucune infrastructure |
| Package API `credit_scoring` v1 | Créé et publié sur l'API Deployer |
| Bundle `bundle_v1` | Impossible : `Bundles are not enabled in your license` |
| Nœud API | Impossible : licence `apiNodeAllowed = false`, et l'installeur Windows ne crée que des nœuds Design |
| Nœud Automation | Impossible pour les mêmes raisons |

## Étapes restantes sur Dataiku Cloud (essai)

1. Exporter le projet (Actions > Export, sans les données) et l'importer sur le Design node Cloud ; recharger les Parquet dans le stockage géré et reconstruire le Flow.
2. Créer une connexion SQL (Snowflake ou PostgreSQL de l'essai) et la choisir dans Administration > Settings > API Designer > « Connection for bundled data ». Relancer les requêtes de test de `predict_default_by_id`.
3. Lancer `python scripts/setup_mlops.py --bundle` : création de `bundle_v1` et publication sur le Project Deployer, publication du package `credit_scoring` v1 sur l'API Deployer.
4. Project Deployer : New deployment > `bundle_v1` > infrastructure Automation fournie par l'essai, puis Deploy. Vérifier que le scénario de monitoring est actif sur le nœud Automation.
5. API Deployer : New deployment > `credit_scoring` v1 > infrastructure API (nœud managé Cloud), puis Deploy. Tester l'URL publique avec `{"features": {"case_id": 33494}}`.
6. Monitoring : dans le déploiement API, activer le logging des requêtes vers l'Event Server, puis créer depuis l'API Deployer la boucle de monitoring (dataset de logs + Evaluate recipe vers un MES), et ajouter l'ensemble dans Unified Monitoring.
