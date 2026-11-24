# Agents (zone 8 Agents)

Trois outils d'agent et deux agents visuels construits avec l'Agent Builder de Dataiku DSS 15, sur le LLM local `custom:ollama:qwen25` (qwen2.5 7B via le plugin `ollama-llm`). Tout est recréé par :

```
python scripts/setup_agents.py --build   # recettes, outils, agents ; --build reconstruit les datasets
python scripts/setup_agents.py --test    # pose les questions de test aux deux agents
```

## Données préparées pour les outils

| Recette | Entrée | Sortie | Contenu |
| --- | --- | --- | --- |
| `case_lookup` (Python) | `hc_prod_scored` | `hc_case_profile` | Un dossier de production par ligne : une vingtaine de colonnes renommées en français (montant, mensualité, revenu, dettes, retards, refus, contrats au bureau), `proba_defaut`, `taux_effort`, `decile_risque`, `classe_risque` (déciles 1 à 5 faible, 6 à 9 modéré, 10 élevé) et `donnees_manquantes` |
| `complaint_stats` (Python) | `cfpb_triage` | `cfpb_stats` | Une ligne par produit (plus `tous`) : nombre de réclamations, répartition par urgence, montant moyen, motifs fréquents |

Le profil allégé évite d'envoyer 500 variables techniques au LLM. La colonne `donnees_manquantes` existe parce que l'outil Dataset Lookup omet les champs vides : sans elle, le modèle inventait un revenu et un taux d'effort.

## Outils

| Outil | Type | Rôle |
| --- | --- | --- |
| `recherche_reglementaire` | Knowledge Bank Search sur `kb_reglementaire` | 4 passages, avec les métadonnées `title` et `page` renvoyées au LLM (paramètre *Columns to retrieve*) pour la citation |
| `profil_dossier` | Dataset Lookup sur `hc_case_profile` | Recherche par `case_id` (seule colonne de recherche autorisée) |
| `stats_reclamations` | Dataset Lookup sur `cfpb_stats` | Recherche par `produit` (prêt immobilier, carte de crédit, compte courant, prêt personnel, prêt auto, autre, tous) |

Les trois outils sont natifs (aucun outil Python). Leur description pour le LLM (*Additional description*) précise la colonne et l'opérateur à utiliser, ce qui suffit à qwen2.5 pour construire le filtre.

## Agent simple : `copilote_analyste`

Agent visuel simple (tools-using agent), température 0, trois outils. Instructions : répondre en français, appeler l'outil adapté au type de question, citer les sources réglementaires sous la forme [titre du document, page X], ne jamais inventer de chiffre, signaler les champs de `donnees_manquantes`.

## Agent structuré : `revue_dossier`

Agent visuel structuré à quatre blocs, déclenché par un message contenant un `case_id` :

1. `extraction` (LLM request, sortie enregistrée dans l'état `case_id`) : le LLM renvoie uniquement l'identifiant.
2. `profil` (Manual tool call) : appel de `profil_dossier` avec le filtre CEL `{"column": "case_id", "operator": "EQUALS", "value": trim(string(state.case_id))}`.
3. `exigences` (Manual tool call) : appel de `recherche_reglementaire` avec une requête fixe sur l'évaluation de solvabilité EBA (en anglais, mieux classée que la version française).
4. `revue` (LLM request, sortie streamée) : revue de conformité en trois parties (profil, exigences citées, avis).

Un bloc *Mandatory tool call* a été testé en premier pour les étapes 1 et 2 : il échoue (« LLM did not emit any tool call ») car Ollama ignore le paramètre `toolChoice`. Le découpage extraction LLM puis appel manuel rend l'enchaînement déterministe.

## Tests (24 novembre 2026, dossier 33602 classé élevé)

| Question | Réponse observée |
| --- | --- |
| Quel est le profil de risque du dossier 33602 ? | « Classe de risque : Élevé, Probabilité de défaut : 69.2%, Montant du crédit : 16 000, Mensualité : 1 961, Retards max. 12 mois : 0 [...] Les informations sur le revenu principal, le taux d'effort [...] sont inconnues. » |
| Le scoring de crédit est-il un système à haut risque selon l'AI Act ? | « les systèmes d'IA utilisés pour évaluer la note de crédit ou la solvabilité des personnes physiques sont classés comme des systèmes d'IA à haut risque [...] Sources : Règlement (UE) 2024/1689 (AI Act), pages 14, 15, 16, 53 » |
| Combien de réclamations urgentes concernent les cartes de crédit ? | « 41 réclamations concernant les cartes de crédit, dont 17 sont d'urgence haute, 3 d'urgence moyenne et 21 d'urgence basse. » (valeurs exactes de `cfpb_stats`) |
| `revue_dossier` : Revue du dossier 33602 | Profil exact (taux d'effort « Non renseigné »), deux exigences citées [EBA/GL/2020/06, pages 9 et 33], avis : vérifier revenu, taux d'effort, emploi et engagements |

## Limites observées

- Le modèle 7B ajoute parfois le symbole € alors que les montants Home Credit sont sans devise, et traduit *borrower* par « prêteur » dans la revue.
- « Urgentes » est interprété comme la répartition complète par urgence : la réponse est juste mais moins directe qu'un seul chiffre.
- Les citations portent sur les passages retournés ; la pertinence dépend du classement de la Knowledge Bank (seuil de similarité 0,5, 4 passages).
- `proba_defaut` est une probabilité calibrée (calibration isotonique depuis le modèle v2) ; la classe de risque relative (décile) reste plus parlante pour l'analyste.
- Chaque réponse prend 5 à 30 secondes sur le GPU local.
