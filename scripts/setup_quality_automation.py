"""Quality gate and automation layer of the CreditOps 360 project:

- zone 2 quality gate: Python recipe validate_base (hc_base -> hc_base_validated),
  and join_features rewired to read hc_base_validated
- Data Quality rules on hc_base_validated and hc_features
- custom Python metric (Gini, max weekly PSI) and check on hc_prod_scored
- step-based scenario weekly_retrain (triggers left inactive)

Idempotent: objects are created when missing and updated otherwise.

Usage (with the Python environment shipped with DSS, which includes dataikuapi):
    python scripts/setup_quality_automation.py [--host http://localhost:11200] [--compute]
--compute builds hc_base_validated and computes the rules and metrics once.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dataikuapi

REPO = Path(__file__).resolve().parents[1]
PROJECT_KEY = "CREDITOPS360"
QUALITY_ZONE = "2 Qualité"
GINI_MIN = 0.55
RECORDS_RANGE = (1_500_000, 1_600_000)


def api_key() -> str:
    key = os.environ.get("DSS_API_KEY")
    if not key:
        f = Path.home() / ".creditops_dss_key"
        key = f.read_text().strip() if f.exists() else ""
    if not key:
        sys.exit("No API key: set DSS_API_KEY or create ~/.creditops_dss_key")
    return key


def source(*parts: str) -> str:
    return (REPO / "dataiku" / Path(*parts)).read_text(encoding="utf-8")


def zone(project, name):
    return next(z for z in project.get_flow().list_zones() if z.name == name)


# ---------------------------------------------------------------- quality gate

def ensure_validate_base(project) -> None:
    code = source("recipes", "validate_base.py")
    if "validate_base" in {r["name"] for r in project.list_recipes()}:
        settings = project.get_recipe("validate_base").get_settings()
        settings.set_code(code)
        settings.save()
        print("recipe validate_base: code updated")
        return
    builder = project.new_recipe("python", "validate_base")
    builder.with_input("hc_base")
    builder.with_new_output_dataset("hc_base_validated", "filesystem_managed")
    builder.with_script(code)
    recipe = builder.create()
    z = zone(project, QUALITY_ZONE)
    z.add_item(recipe)
    z.add_item(project.get_dataset("hc_base_validated"))
    print("recipe validate_base: created")


def rewire_join_features(project) -> None:
    """join_features reads the base table from hc_base_validated (static tables stay in the folder)."""
    settings = project.get_recipe("join_features").get_settings()
    settings.set_code(source("recipes", "join_features.py"))
    if not settings.has_input("hc_base_validated"):
        settings.add_input("main", "hc_base_validated")
        print("recipe join_features: input hc_base_validated added")
    settings.save()
    print("recipe join_features: code updated")


def apply_split_schema(project) -> None:
    updates = project.get_recipe("split_train_prod").compute_schema_updates()
    if updates.any_action_required():
        updates.apply()
        print("recipe split_train_prod: output schemas updated")


# ---------------------------------------------------------- data quality rules

def dq_rules(n_min: int, n_max: int) -> list[dict]:
    return [
        {"type": "ColumnUniqueValuesRule", "displayName": "case_id unique",
         "columns": ["case_id"], "thresholdType": "ENTIRE_COLUMN"},
        {"type": "ColumnNotEmptyRule", "displayName": "target not empty",
         "columns": ["target"], "thresholdType": "ENTIRE_COLUMN_NOT_EMPTY"},
        {"type": "RecordCountInRangeRule", "displayName": "record count 1.5M-1.6M",
         "minimum": n_min, "minimumEnabled": True, "maximum": n_max, "maximumEnabled": True},
        {"type": "ValuesInSetRule", "displayName": "target in {0,1}",
         "columns": ["target"], "valueSet": ["0", "1"]},
    ]


def sync_rules(dataset, rules: list[dict]) -> None:
    """Creates or updates rules, matched on their display name."""
    ruleset = dataset.get_data_quality_rules()
    existing = {r.name: r for r in ruleset.list_rules()}
    for config in rules:
        rule = existing.get(config["displayName"])
        if rule is None:
            ruleset.create_rule({**config, "enabled": True, "autoRun": True})
        else:
            rule.get_raw().update(config)
            rule.save()
    print(f"dataset {dataset.dataset_name}: {len(rules)} data quality rules")


# ------------------------------------------------------ custom metric + check

PROBE_NAME = "model_quality"
METRICS = ("gini_overall", "psi_weekly_max", "weeks_scored")
GINI_METRIC = f"python:gini_overall:{PROBE_NAME}"  # python:<returned key>:<probe name>


def ensure_model_quality_metric(project) -> None:
    ds = project.get_dataset("hc_prod_scored")
    settings = ds.get_settings()
    metrics = settings.get_raw()["metrics"]
    probes = [p for p in metrics["probes"] if p.get("meta", {}).get("name") != PROBE_NAME]
    probes.append({
        "type": "python", "enabled": True, "computeOnBuildMode": "NO",
        "meta": {"name": PROBE_NAME, "level": 2},
        "configuration": {"code": source("metrics", "model_quality_probe.py"),
                          "envSelection": {"envMode": "INHERIT"}},
    })
    metrics["probes"] = probes
    shown = metrics.setdefault("displayedState", {}).setdefault("metrics", [])
    shown += [f"python:{m}:{PROBE_NAME}" for m in METRICS if f"python:{m}:{PROBE_NAME}" not in shown]
    settings.save()

    # Drop malformed legacy checks (no metric) left by earlier attempts
    ruleset = ds.get_data_quality_rules()
    for rule in ruleset.list_rules():
        if rule.get_raw()["type"] in ("python", "numericRange") and not rule.get_raw().get("metricId") \
                and rule.name in ("Custom check", "Metric in range (numeric) of null"):
            rule.delete()
    sync_rules(ds, [{
        "type": "numericRange", "displayName": f"Gini >= {GINI_MIN}",
        "metricId": GINI_METRIC, "minimum": GINI_MIN, "minimumEnabled": True,
        "autoComputeMetric": False,
    }])
    print("dataset hc_prod_scored: custom metric and Gini check")


# ------------------------------------------------------------------- scenario

SCENARIO_NAME = "weekly_retrain"


def build_step(name: str, items: list[tuple[str, str]], mode: str, update_schemas: bool = False) -> dict:
    return {
        "type": "build_flowitem", "name": name, "enabled": True,
        "runConditionType": "RUN_IF_STATUS_MATCH", "runConditionStatuses": ["SUCCESS", "WARNING"],
        "params": {"jobType": mode, "refreshHiveMetastore": False, "proceedOnFailure": False,
                   "autoUpdateSchemaBeforeEachRecipeRun": update_schemas,
                   "builds": [{"type": t, "itemId": i, "partitionsSpec": ""} for t, i in items]},
    }


def scenario_steps(project_key: str) -> list[dict]:
    dq_items = [{"type": "DATASET", "projectKey": project_key, "itemId": d, "partitionsSpec": ""}
                for d in ("hc_base_validated",)]
    return [
        build_step("Build hc_base_validated (quality gate)", [("DATASET", "hc_base_validated")],
                   "NON_RECURSIVE_FORCED_BUILD"),
        {"type": "check_dataset", "name": "Data quality rules (stop on error)", "enabled": True,
         "runConditionType": "RUN_IF_STATUS_MATCH", "runConditionStatuses": ["SUCCESS", "WARNING"],
         "params": {"checks": dq_items, "computeAutomaticRules": True,
                    "handleWarningsAs": "WARNING"}},
        # One forced, non recursive build per stage, in flow order. A single job with all
        # items runs recipes in parallel and a reader can open hc_features while it is
        # still being written (file lock on Windows). Non recursive: pd_model is never
        # retrained here.
        *[build_step("Build " + " + ".join(group), [("DATASET", d) for d in group],
                     "NON_RECURSIVE_FORCED_BUILD", update_schemas=True)
          for group in (("hc_features",), ("hc_train", "hc_prod"), ("hc_prod_scored",),
                        ("hc_weekly_monitoring",), ("hc_stability_summary",))],
        {"type": "custom_python", "name": "Retrain pd_model on alert", "enabled": True,
         "runConditionType": "RUN_IF_STATUS_MATCH", "runConditionStatuses": ["SUCCESS", "WARNING"],
         "params": {"script": source("scenarios", "weekly_retrain_step.py"),
                    "envSelection": {"envMode": "INHERIT"}}},
    ]


def ensure_scenario(project):
    existing = {s["name"]: s["id"] for s in project.list_scenarios()}
    if SCENARIO_NAME in existing:
        scenario = project.get_scenario(existing[SCENARIO_NAME])
    else:
        scenario = project.create_scenario(SCENARIO_NAME, "step_based")
        print(f"scenario {SCENARIO_NAME}: created")
    settings = scenario.get_settings()
    raw = settings.get_raw()
    raw["params"]["steps"] = scenario_steps(project.project_key)
    raw["description"] = ("Weekly run: quality gate, data quality rules, scoring of the new weeks, "
                          "monitoring, retraining of pd_model only when PSI or Gini alerts are raised.")
    if not any(t.get("type") == "temporal" for t in raw.get("triggers", [])):
        settings.add_daily_trigger(hour=6, minute=0, days=["Monday"])
    for t in raw["triggers"]:
        t["active"] = False  # schedule defined but not activated
    settings.active = False
    settings.save()
    print(f"scenario {SCENARIO_NAME}: {len(raw['params']['steps'])} steps, triggers inactive")
    return scenario


# ---------------------------------------------------------------------- main

def compute(project) -> None:
    """Builds the quality gate output and computes rules and metrics once."""
    job = project.new_job("NON_RECURSIVE_FORCED_BUILD")
    job.with_output("hc_base_validated")
    job.start_and_wait()
    print("hc_base_validated: built")
    for name in ("hc_base_validated", "hc_features"):
        project.get_dataset(name).get_data_quality_rules().compute_rules().wait_for_result()
    ds = project.get_dataset("hc_prod_scored")
    ds.compute_metrics()
    ds.get_data_quality_rules().compute_rules().wait_for_result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11200")
    parser.add_argument("--compute", action="store_true")
    args = parser.parse_args()

    project = dataikuapi.DSSClient(args.host, api_key()).get_project(PROJECT_KEY)
    ensure_validate_base(project)
    rewire_join_features(project)
    apply_split_schema(project)
    n_min, n_max = RECORDS_RANGE
    for name in ("hc_base_validated", "hc_features"):
        sync_rules(project.get_dataset(name), dq_rules(n_min, n_max))
    ensure_model_quality_metric(project)
    ensure_scenario(project)
    if args.compute:
        compute(project)
    print("done")


if __name__ == "__main__":
    main()
