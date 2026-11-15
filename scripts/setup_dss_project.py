"""Creates (or updates) the CreditOps 360 project skeleton on a DSS instance:
connection, project, zones, variables, raw folder, project library, ingestion recipes
and source datasets.

Idempotent: re-running it only adds what is missing.

Usage (with the Python environment shipped with DSS, which includes dataikuapi):
    set DSS_API_KEY=<personal API key>
    python scripts/setup_dss_project.py [--host http://localhost:11200]
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile
import re
import sys
from pathlib import Path

import dataikuapi
from dataikuapi.dss.recipe import EmbedDatasetRecipeCreator

REPO = Path(__file__).resolve().parents[1]
CONNECTION = "creditops_data"
PROJECT_KEY = "CREDITOPS360"
PROJECT_NAME = "CreditOps 360"
ZONES = ["1 Ingestion", "2 Qualité", "3 Features", "4 ML", "5 Monitoring",
         "6 Réclamations", "7 Réglementaire", "8 Agents"]
VARIABLES = {"train_week_max": 40, "psi_threshold": 0.2, "gini_drop_threshold": 0.05}
HC_TRAIN = "homecredit/parquet_files/train"


def api_key() -> str:
    key = os.environ.get("DSS_API_KEY")
    if not key:
        f = Path.home() / ".creditops_dss_key"
        key = f.read_text().strip() if f.exists() else ""
    if not key:
        sys.exit("No API key: set DSS_API_KEY or create ~/.creditops_dss_key")
    return key


def ensure_connection(client) -> None:
    root = str(REPO / "data").replace("\\", "/")
    if CONNECTION in client.list_connections():
        print(f"connection {CONNECTION}: exists")
        return
    client.create_connection(CONNECTION, type="Filesystem", params={"root": root})
    print(f"connection {CONNECTION}: created -> {root}")


def ensure_project(client):
    if PROJECT_KEY not in client.list_project_keys():
        client.create_project(PROJECT_KEY, PROJECT_NAME, owner=client.get_auth_info()["authIdentifier"])
        print(f"project {PROJECT_KEY}: created")
    project = client.get_project(PROJECT_KEY)
    variables = project.get_variables()
    variables["standard"].update(VARIABLES)
    project.set_variables(variables)
    return project


def ensure_zones(project) -> dict:
    flow = project.get_flow()
    zones = {z.name: z for z in flow.list_zones()}
    for name in ZONES:
        if name not in zones:
            zones[name] = flow.create_zone(name)
            print(f"zone {name}: created")
    return zones


def files_rule(pattern: str) -> dict:
    return {"mode": "RULES_INCLUDED_ONLY", "explicitFiles": [], "excludeRules": [],
            "includeRules": [{"mode": "GLOB", "matchingMode": "FILENAME", "expr": pattern}]}


def ensure_dataset(project, zone, name, path, pattern, fmt, fmt_params=None):
    existing = {d["name"] for d in project.list_datasets()}
    if name in existing:
        return
    ds = project.create_dataset(
        name, type="Filesystem",
        params={"connection": CONNECTION, "path": path, "filesSelectionRules": files_rule(pattern)},
        formatType=fmt, formatParams=fmt_params or {},
    )
    settings = ds.autodetect_settings()
    settings.save()
    zone.add_item(ds)
    print(f"dataset {name}: created")


LIB_SRC = REPO / "dataiku" / "python" / "creditops"
RECIPES = {
    # recipe name: (zone, inputs, outputs); inputs may be datasets or FOLDERS
    "ingest_homecredit": ("1 Ingestion", ["hc_raw_parquet"], [f"hc_{t}" for t in (
        "base", "static_0", "static_cb_0", "applprev_1", "person_1", "credit_bureau_b_1",
        "debitcard_1", "deposit_1", "other_1", "tax_registry_a_1", "tax_registry_b_1",
        "tax_registry_c_1")]),
    "aggregate_bureau": ("3 Features", ["hc_raw_parquet"], ["hc_agg_bureau"]),
    "chunk_regulations": ("7 Réglementaire", ["reg_documents"], ["reg_chunks"]),
    "join_features": ("3 Features", ["hc_raw_parquet", "hc_agg_bureau"], ["hc_features"]),
    "prepare_complaints": ("6 Réclamations", ["cfpb_complaints_raw"], ["cfpb_sample"]),
    # needs hc_prod_scored, produced by the visual Score recipe on the deployed model
    "weekly_monitoring": ("5 Monitoring", ["hc_prod_scored"],
                          ["hc_weekly_monitoring", "hc_stability_summary"]),
}


FOLDERS = {
    # folder name: (zone, path in the creditops_data connection)
    "hc_raw_parquet": ("1 Ingestion", "/raw/" + HC_TRAIN),  # DSS cannot read Parquet natively on Windows
    "reg_documents": ("7 Réglementaire", "/raw/reglementaire"),
}
CODE_ENV = "creditops"
CODE_ENV_PACKAGES = "pypdf==5.1.0\nscikit-learn\n"
RECIPE_CODE_ENV = {"chunk_regulations": CODE_ENV}


def ensure_folders(project, zones) -> dict:
    """Folders pointing at raw files; returns folder name -> id."""
    ids = {f["name"]: f["id"] for f in project.list_managed_folders()}
    for name, (zone, path) in FOLDERS.items():
        if name in ids:
            continue
        folder = project.create_managed_folder(name, connection_name=CONNECTION)
        settings = folder.get_settings()
        settings.get_raw()["params"]["path"] = path
        settings.save()
        zones[zone].add_item(folder)
        ids[name] = folder.id
        print(f"folder {name}: created")
    return ids


def ensure_code_env(client) -> None:
    if CODE_ENV not in {e["envName"] for e in client.list_code_envs()}:
        client.create_code_env("PYTHON", CODE_ENV, "DESIGN_MANAGED",
                               {"pythonInterpreter": "PYTHON311", "installCorePackages": True})
        env = client.get_code_env("PYTHON", CODE_ENV)
        settings = env.get_settings()
        settings.get_raw()["specPackageList"] = CODE_ENV_PACKAGES
        settings.save()
        env.update_packages()
        print(f"code env {CODE_ENV}: created")


def sync_library(project) -> None:
    """Pushes dataiku/python/creditops into the project library."""
    lib = project.get_library()
    for src in sorted(LIB_SRC.glob("*.py")):
        path = f"python/creditops/{src.name}"
        try:
            lib.get_file(path).write(src.read_text(encoding="utf-8"))
        except Exception:
            lib.add_file(path).write(src.read_text(encoding="utf-8"))
    print("library: synced")


def ensure_recipes(project, zones, folder_ids) -> None:
    existing = {r["name"] for r in project.list_recipes()}
    for name, (zone, inputs, outputs) in RECIPES.items():
        code = (REPO / "dataiku" / "recipes" / f"{name}.py").read_text(encoding="utf-8")
        if name in existing:
            settings = project.get_recipe(name).get_settings()
            settings.set_code(code)
            settings.save()
            continue
        builder = project.new_recipe("python", name)
        datasets = {d["name"] for d in project.list_datasets()}
        if any(i not in folder_ids and i not in datasets for i in inputs):
            print(f"recipe {name}: skipped, missing input")
            continue
        for inp in inputs:
            builder.with_input(folder_ids.get(inp, inp))
        for out in outputs:
            builder.with_new_output_dataset(out, "filesystem_managed")
        builder.with_script(code)
        recipe = builder.create()
        if name in RECIPE_CODE_ENV:
            settings = recipe.get_settings()
            settings.get_recipe_raw_definition()["params"]["envSelection"] = {
                "envMode": "EXPLICIT_ENV", "envName": RECIPE_CODE_ENV[name]}
            settings.save()
        zones[zone].add_item(recipe)
        for out in outputs:
            zones[zone].add_item(project.get_dataset(out))
        print(f"recipe {name}: created")


def ensure_split(project, zone) -> None:
    """Visual Split recipe: training weeks vs simulated production weeks."""
    if "split_train_prod" in {r["name"] for r in project.list_recipes()}:
        return
    builder = project.new_recipe("split", "split_train_prod")
    builder.with_input("hc_features")
    builder.with_new_output("hc_train", "filesystem_managed")
    builder.with_new_output("hc_prod", "filesystem_managed")
    recipe = builder.create()
    settings = recipe.get_settings()
    settings.set_json_payload({
        "mode": "FILTERS",
        "filterSplits": [{
            "outputIndex": 0,
            "filter": {"enabled": True, "uiData": {"mode": "CUSTOM"},
                       "expression": "WEEK_NUM <= ${train_week_max}"},
        }],
        "defaultOutputIndex": 1,
    })
    settings.save()
    updates = recipe.compute_schema_updates()  # propagates the input schema to both outputs
    if updates.any_action_required():
        updates.apply()
    for item in (recipe, project.get_dataset("hc_train"), project.get_dataset("hc_prod")):
        zone.add_item(item)
    print("recipe split_train_prod: created")


PLUGIN_DIR = REPO / "dataiku" / "plugins" / "ollama-llm"
LLM_CONNECTION = "ollama"
LLM_MODELS = [
    # (id in the LLM Mesh, capability, plugin component, component settings)
    ("qwen25", "TEXT_COMPLETION", "ollama-chat",
     {"base_url": "http://localhost:11434", "model": "qwen2.5:7b", "num_ctx": 8192, "timeout": 600,
      "system_prompt": "Tu es un analyste bancaire. Réponds uniquement en français."}),
    ("bge-m3", "TEXT_EMBEDDING", "ollama-embedding",
     {"base_url": "http://localhost:11434", "model": "bge-m3", "timeout": 120}),
]


def ensure_plugin(client) -> None:
    """Installs or updates the Ollama LLM plugin from the repository sources."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for f in PLUGIN_DIR.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, f.relative_to(PLUGIN_DIR).as_posix())
    buffer.seek(0)
    if "ollama-llm" in {p["id"] for p in client.list_plugins()}:
        client.get_plugin("ollama-llm").update_from_zip(buffer)
    else:
        client.install_plugin_from_archive(buffer)
    print("plugin ollama-llm: installed")


def ensure_llm_connection(client) -> None:
    """Custom LLM connection exposing the local Ollama models to the LLM Mesh."""
    if LLM_CONNECTION not in client.list_connections():
        client.create_connection(LLM_CONNECTION, type="CustomLLM", params={})
    conn = client.get_connection(LLM_CONNECTION)
    definition = conn.get_definition()
    definition["params"]["pluginId"] = "ollama-llm"
    definition["params"]["models"] = [{
        "id": model_id, "capability": capability,
        "type": f"llm_ollama-llm_{component}", "customConfig": config,
        "maxParallelism": 2,  # a single local GPU
    } for model_id, capability, component, config in LLM_MODELS]
    conn.set_definition(definition)
    print(f"connection {LLM_CONNECTION}: models {[m[0] for m in LLM_MODELS]}")


def ensure_knowledge_bank(project, zones) -> None:
    """Embed recipe: regulatory passages -> Chroma Knowledge Bank (bge-m3 through Ollama).

    Needs the DSS internal "Retrieval augmented generation" code env
    (Administration > Code envs > Internal envs setup).
    """
    if "embed_regulations" in {r["name"] for r in project.list_recipes()} \
            or "reg_chunks" not in {d["name"] for d in project.list_datasets()}:
        return
    builder = EmbedDatasetRecipeCreator("embed_regulations", project)
    builder.with_input("reg_chunks")
    builder.with_output_knowledge_bank("kb_reglementaire", f"custom:{LLM_CONNECTION}:bge-m3", "CHROMA")
    recipe = builder.create()
    settings = recipe.get_settings()
    settings.embedding_column = "text"
    payload = settings.get_json_payload()
    payload["metadataColumns"] = [{"column": c} for c in ("chunk_id", "source", "title", "page", "language")]
    settings.set_json_payload(payload)
    settings.save()
    zones["7 Réglementaire"].add_item(recipe)
    print("knowledge bank kb_reglementaire: created")


def drop_parquet_datasets(project) -> None:
    """Removes datasets from an earlier version that pointed DSS directly at Parquet files."""
    for d in project.list_datasets():
        if d["name"].startswith("hc_") and d["name"] != "hc_feature_definitions" \
                and d.get("params", {}).get("connection") == CONNECTION:
            project.get_dataset(d["name"]).delete()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11200")
    args = parser.parse_args()

    client = dataikuapi.DSSClient(args.host, api_key())
    ensure_connection(client)
    ensure_plugin(client)
    ensure_llm_connection(client)
    project = ensure_project(client)
    zones = ensure_zones(project)

    ingestion = zones["1 Ingestion"]
    drop_parquet_datasets(project)
    ensure_code_env(client)
    folder_ids = ensure_folders(project, zones)
    sync_library(project)
    ensure_recipes(project, zones, folder_ids)
    ensure_split(project, zones["4 ML"])
    ensure_knowledge_bank(project, zones)

    csv = {"style": "excel", "charset": "utf-8", "separator": ",", "quoteChar": '"',
           "escapeChar": "\\", "parseHeaderRow": True, "arrayMapFormat": "json"}
    ensure_dataset(project, ingestion, "hc_feature_definitions", "raw/homecredit",
                   "feature_definitions.csv", "csv", csv)
    ensure_dataset(project, zones["6 Réclamations"], "cfpb_complaints_raw", "raw/cfpb",
                   "CCDB_Export_*.csv", "csv", csv)
    print("done")


if __name__ == "__main__":
    main()
