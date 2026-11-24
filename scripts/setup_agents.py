"""Creates (or updates) the Agent Builder objects of CreditOps 360 in zone "8 Agents":

- recipes case_lookup (hc_prod_scored -> hc_case_profile) and complaint_stats
  (cfpb_triage -> cfpb_stats), whose outputs feed the Dataset Lookup tools;
- agent tools recherche_reglementaire (Knowledge Bank search on kb_reglementaire),
  profil_dossier and stats_reclamations (Dataset Lookup);
- simple visual agent copilote_analyste (tools-using agent on qwen2.5 via Ollama);
- structured visual agent revue_dossier (blocks: profile lookup -> regulatory
  retrieval -> written compliance review).

Idempotent: objects are looked up by name and their settings are rewritten.

Usage (with the Python environment shipped with DSS, which includes dataikuapi):
    python scripts/setup_agents.py [--host http://localhost:11200] [--build] [--test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dataikuapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from setup_dss_project import PROJECT_KEY, REPO, api_key  # noqa: E402

ZONE = "8 Agents"
LLM_ID = "custom:ollama:qwen25"
KB_NAME = "kb_reglementaire"
RECIPES = {
    # recipe name: (inputs, outputs)
    "case_lookup": (["hc_prod_scored"], ["hc_case_profile"]),
    "complaint_stats": (["cfpb_triage"], ["cfpb_stats"]),
}
PRODUCTS = "prêt immobilier, carte de crédit, compte courant, prêt personnel, prêt auto, autre, tous"

TOOLS = {
    "recherche_reglementaire": {
        "type": "VectorStoreSearch",
        "description": (
            "Recherche des passages dans les textes réglementaires : orientations EBA sur l'octroi "
            "et le suivi des prêts (EBA/GL/2020/06), AI Act (règlement UE 2024/1689), directive "
            "crédit à la consommation 2023/2225, recommandation ACPR. Chaque passage indique sa "
            "source (title) et sa page. Utilise des mots-clés précis en français ou en anglais."),
    },
    "profil_dossier": {
        "type": "DatasetRowLookup",
        "dataset": "hc_case_profile",
        "lookup": ["case_id"],
        "description": (
            "Renvoie le profil d'un dossier de crédit à partir de son identifiant case_id "
            "(nombre entier, ex. 1234567) : montant, mensualité, revenu, taux d'effort, retards, "
            "probabilité de défaut (proba_defaut) et classe de risque (faible, modéré, élevé). "
            "Filtre : column = case_id, operator = EQUALS, value = l'identifiant."),
    },
    "stats_reclamations": {
        "type": "DatasetRowLookup",
        "dataset": "cfpb_stats",
        "lookup": ["produit"],
        "description": (
            "Statistiques des réclamations clients (base CFPB classée par LLM) pour un produit : "
            "nombre total, nombre par urgence (haute, moyenne, basse), montant moyen, motifs "
            "fréquents. Filtre : column = produit, operator = EQUALS, value = un de : "
            + PRODUCTS + " (\"tous\" = toutes les réclamations)."),
    },
}

COPILOT_PROMPT = """Tu es le copilote d'un analyste crédit dans une banque de détail française. Réponds en français, de façon concise.
Règles :
1. Question sur un dossier ou un case_id : appelle profil_dossier, puis résume la classe de risque, la probabilité de défaut, le montant, la mensualité, le taux d'effort et les retards, sans citer de source.
2. Question réglementaire (EBA, AI Act, directive crédit, ACPR) : appelle recherche_reglementaire. Termine ta réponse par la liste des sources utilisées, sous la forme [titre du document, page X].
3. Question sur les réclamations clients : appelle stats_reclamations avec le produit (ex. carte de crédit) ou tous.
4. N'invente jamais de chiffre : utilise uniquement les valeurs renvoyées par les outils. Les montants des dossiers sont sans devise. Les champs listés dans donnees_manquantes sont inconnus : dis-le."""

REVIEW_EXTRACT_PROMPT = (
    "Donne uniquement l'identifiant de dossier (case_id) présent dans le message de "
    "l'utilisateur, en chiffres, sans aucun autre texte. Exemple de réponse : 1234567")
# CEL expression evaluated by the block: Dataset Lookup filter on the extracted id
REVIEW_LOOKUP_FILTER = '{"column": "case_id", "operator": "EQUALS", "value": trim(string(state.case_id))}'
# English query: the EBA guidelines are indexed in English and French, the English
# wording retrieves the creditworthiness section more reliably
REVIEW_KB_QUERY = ("'EBA guidelines loan origination creditworthiness assessment of consumers: "
                   "income, expenses, debt service, verification of information'")
REVIEW_PROMPT = """Tu es un contrôleur conformité crédit. À partir du profil du dossier et des extraits réglementaires EBA ci-dessus, rédige en français une revue de conformité courte (150 mots maximum) avec trois parties :
1. Profil : classe de risque, probabilité de défaut, montant, mensualité, taux d'effort, retards (valeurs exactes de l'outil, montants sans devise).
2. Exigences réglementaires : deux ou trois points, chacun cité sous la forme [titre du document, page X].
3. Avis : points à vérifier avant décision.
Les champs listés dans donnees_manquantes sont inconnus : écris "non renseigné" et ne les calcule pas.
N'invente aucun chiffre ni aucune source."""

TEST_QUESTIONS = [
    "Quel est le profil de risque du dossier {case_id} ?",
    "Le scoring de crédit est-il un système à haut risque selon l'AI Act ?",
    "Combien de réclamations urgentes concernent les cartes de crédit ?",
]


def ensure_recipes(project, zone) -> None:
    existing = {r["name"] for r in project.list_recipes()}
    for name, (inputs, outputs) in RECIPES.items():
        code = (REPO / "dataiku" / "recipes" / f"{name}.py").read_text(encoding="utf-8")
        if name in existing:
            settings = project.get_recipe(name).get_settings()
            settings.set_code(code)
            settings.save()
            continue
        builder = project.new_recipe("python", name)
        for inp in inputs:
            builder.with_input(inp)
        for out in outputs:
            builder.with_new_output_dataset(out, "filesystem_managed")
        builder.with_script(code)
        recipe = builder.create()
        zone.add_item(recipe)
        for out in outputs:
            zone.add_item(project.get_dataset(out))
        print(f"recipe {name}: created")


def build_outputs(project) -> None:
    for _, outputs in RECIPES.values():
        for out in outputs:
            project.get_dataset(out).build(job_type="NON_RECURSIVE_FORCED_BUILD", wait=True)
            print(f"dataset {out}: built")


def ensure_tool(project, name, spec, kb_id) -> str:
    tools = {t.name: t.id for t in project.list_agent_tools()}
    if name in tools:
        tool = project.get_agent_tool(tools[name])
    else:
        creator = project.new_agent_tool(spec["type"], name=name)
        if spec["type"] == "VectorStoreSearch":
            creator.with_knowledge_bank(kb_id)
        tool = creator.create()
        print(f"tool {name}: created")
    settings = tool.get_settings()
    raw = settings.get_raw()
    raw["additionalDescriptionForLLM"] = spec["description"]
    params = raw["params"]
    if spec["type"] == "VectorStoreSearch":
        params["knowledgeBankRef"] = kb_id
        params["maxDocuments"] = 4
        # metadata columns returned with each passage, so the LLM can cite document and page
        params["retrievalColumns"] = ["DKU_TEXT_EMBEDDING_COLUMN", "title", "page"]
        params.setdefault("sourcesSettings", {})["metadataInSources"] = ["title", "page", "source"]
    else:
        params["datasetRef"] = spec["dataset"]
        params["usableLookupColumns"] = spec["lookup"]
        params["returnedColumns"] = []
        params["retrievalMode"] = "SINGLE_RECORD"
    settings.save()
    return tool.id


def get_or_create_agent(project, name, agent_type):
    for a in project.list_agents():
        if a.name == name:
            return project.get_agent(a.id)
    print(f"agent {name}: created")
    return project.create_agent(name, agent_type)


def ensure_copilot(project, tool_ids):
    agent = get_or_create_agent(project, "copilote_analyste", "TOOLS_USING_AGENT")
    settings = agent.get_settings()
    version = settings.get_version_settings(settings.active_version)
    tua = version.get_raw()["toolsUsingAgentSettings"]
    tua["llmId"] = LLM_ID
    tua["systemPromptAppend"] = COPILOT_PROMPT
    tua["tools"] = [{"toolRef": tool_ids[n]} for n in TOOLS]
    tua["completionSettings"]["temperature"] = 0
    settings.save()
    return agent


def ensure_review_agent(project, tool_ids):
    agent = get_or_create_agent(project, "revue_dossier", "STRUCTURED_AGENT")
    settings = agent.get_settings()
    version = settings.get_version_settings(settings.active_version)
    sa = version.get_raw()["structuredAgentSettings"]
    sa["blocks"] = [
        {
            # Ollama ignores toolChoice, so a MANDATORY_TOOL_CALL block is not reliable with
            # qwen2.5: the LLM only extracts the id, the lookup itself is a manual tool call.
            "id": "extraction", "type": "LLM_REQUEST",
            "description": "Extrait le case_id du message",
            "llmId": LLM_ID, "passConversationHistory": True, "streamOutput": False,
            "systemPromptAfterHistory": REVIEW_EXTRACT_PROMPT,
            "completionSettings": {"temperature": 0},
            "outputMode": "SAVE_TO_STATE", "outputKey": "case_id", "nextBlock": "profil",
        },
        {
            "id": "profil", "type": "MANUAL_TOOL_CALL",
            "description": "Lit le profil du dossier",
            "tool": {"type": "EXPLICIT_TOOL", "toolRef": tool_ids["profil_dossier"],
                     "setArgs": [{"key": "filter", "value": REVIEW_LOOKUP_FILTER}]},
            "outputMode": "ADD_TO_MESSAGES", "nextBlock": "exigences",
        },
        {
            "id": "exigences", "type": "MANUAL_TOOL_CALL",
            "description": "Recherche les exigences EBA sur l'évaluation de solvabilité",
            "tool": {"type": "EXPLICIT_TOOL", "toolRef": tool_ids["recherche_reglementaire"],
                     "setArgs": [{"key": "searchQuery", "value": REVIEW_KB_QUERY}]},
            "outputMode": "ADD_TO_MESSAGES", "nextBlock": "revue",
        },
        {
            "id": "revue", "type": "LLM_REQUEST",
            "description": "Rédige la revue de conformité",
            "llmId": LLM_ID, "passConversationHistory": True, "streamOutput": True,
            "systemPromptAfterHistory": REVIEW_PROMPT,
            "completionSettings": {"temperature": 0},
            "outputMode": "ADD_TO_MESSAGES",
        },
    ]
    sa["startingBlockId"] = "extraction"
    sa["nextTurnBehaviour"] = "STARTING_BLOCK"
    settings.save()
    return agent


def add_to_zone(project, zone, agent) -> None:
    try:
        zone.add_item(project.get_saved_model(agent.id))
    except Exception as e:  # already in the zone, or not movable
        print(f"zone: {agent.id} not moved ({e})")


def ask(project, agent, question) -> str:
    completion = project.get_llm(f"agent:{agent.id}").new_completion()
    completion.with_message(question)
    resp = completion.execute()
    return resp.text if resp.success else f"ERROR {resp._raw}"


def sample_case_id(project) -> str:
    dataset = project.get_dataset("hc_case_profile")
    columns = [c["name"] for c in dataset.get_schema()["columns"]]
    for values in dataset.iter_rows():
        row = dict(zip(columns, values))
        if row["classe_risque"] == "élevé":
            return row["case_id"]
    raise RuntimeError("no case in hc_case_profile")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11200")
    parser.add_argument("--build", action="store_true", help="build hc_case_profile and cfpb_stats")
    parser.add_argument("--test", action="store_true", help="ask the test questions to both agents")
    args = parser.parse_args()

    client = dataikuapi.DSSClient(args.host, api_key())
    project = client.get_project(PROJECT_KEY)
    zone = {z.name: z for z in project.get_flow().list_zones()}[ZONE]

    ensure_recipes(project, zone)
    if args.build:
        build_outputs(project)

    kb_id = {kb.name: kb.id for kb in project.list_knowledge_banks()}[KB_NAME]
    tool_ids = {name: ensure_tool(project, name, spec, kb_id) for name, spec in TOOLS.items()}
    copilot = ensure_copilot(project, tool_ids)
    review = ensure_review_agent(project, tool_ids)
    for agent in (copilot, review):
        add_to_zone(project, zone, agent)
    print("tools:", tool_ids, "| agents:", copilot.id, review.id)

    if args.test:
        case_id = sample_case_id(project)
        for q in TEST_QUESTIONS:
            q = q.format(case_id=case_id)
            print(f"\n### copilote_analyste: {q}\n{ask(project, copilot, q)}")
        q = f"Revue du dossier {case_id}"
        print(f"\n### revue_dossier: {q}\n{ask(project, review, q)}")


if __name__ == "__main__":
    main()
