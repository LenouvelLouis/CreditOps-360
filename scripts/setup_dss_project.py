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
import os
import re
import sys
from pathlib import Path

import dataikuapi

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
    # recipe name: (zone, extra input datasets, outputs); every recipe also reads hc_raw_parquet
    "ingest_homecredit": ("1 Ingestion", [], [f"hc_{t}" for t in (
        "base", "static_0", "static_cb_0", "applprev_1", "person_1", "credit_bureau_b_1",
        "debitcard_1", "deposit_1", "other_1", "tax_registry_a_1", "tax_registry_b_1",
        "tax_registry_c_1")]),
    "aggregate_bureau": ("3 Features", [], ["hc_agg_bureau"]),
    "join_features": ("3 Features", ["hc_agg_bureau"], ["hc_features"]),
}


def ensure_raw_folder(project, zone):
    """Folder pointing at the raw Parquet files (DSS cannot read Parquet natively on Windows)."""
    for f in project.list_managed_folders():
        if f["name"] == "hc_raw_parquet":
            return f["id"]
    folder = project.create_managed_folder("hc_raw_parquet", connection_name=CONNECTION)
    settings = folder.get_settings()
    settings.get_raw()["params"]["path"] = "/raw/" + HC_TRAIN
    settings.save()
    zone.add_item(folder)
    print("folder hc_raw_parquet: created")
    return folder.id


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


def ensure_recipes(project, zones, folder_id) -> None:
    existing = {r["name"] for r in project.list_recipes()}
    for name, (zone, inputs, outputs) in RECIPES.items():
        code = (REPO / "dataiku" / "recipes" / f"{name}.py").read_text(encoding="utf-8")
        if name in existing:
            settings = project.get_recipe(name).get_settings()
            settings.set_code(code)
            settings.save()
            continue
        builder = project.new_recipe("python", name)
        builder.with_input(folder_id)
        for inp in inputs:
            builder.with_input(inp)
        for out in outputs:
            builder.with_new_output_dataset(out, "filesystem_managed")
        builder.with_script(code)
        recipe = builder.create()
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
    for item in (recipe, project.get_dataset("hc_train"), project.get_dataset("hc_prod")):
        zone.add_item(item)
    print("recipe split_train_prod: created")


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
    project = ensure_project(client)
    zones = ensure_zones(project)

    ingestion = zones["1 Ingestion"]
    drop_parquet_datasets(project)
    folder_id = ensure_raw_folder(project, ingestion)
    sync_library(project)
    ensure_recipes(project, zones, folder_id)
    ensure_split(project, zones["4 ML"])

    csv = {"style": "excel", "charset": "utf-8", "separator": ",", "quoteChar": '"',
           "escapeChar": "\\", "parseHeaderRow": True, "arrayMapFormat": "json"}
    ensure_dataset(project, ingestion, "hc_feature_definitions", "raw/homecredit",
                   "feature_definitions.csv", "csv", csv)
    ensure_dataset(project, zones["6 Réclamations"], "cfpb_complaints_raw", "raw/cfpb",
                   "CCDB_Export_*.csv", "csv", csv)
    print("done")


if __name__ == "__main__":
    main()
