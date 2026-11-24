"""MLOps layer of CreditOps 360 (zone "5 Monitoring" + API service + bundle).

Creates, if missing:
  * hc_prod_recent      : Filter/Sample recipe on hc_prod (recent production weeks)
  * hc_enrich_cases     : small table of production cases used for API query enrichment
  * mes_pd_model        : Model Evaluation Store fed by the Evaluate recipe evaluate_pd_model
  * mes_pd_model_metrics: metrics dataset (one row per evaluation)
  * credit_scoring      : API service on pd_model, endpoints predict_default (full feature
                          vector) and predict_default_by_id (case_id only, query enrichment
                          on hc_enrich_cases)
  * bundle_v1           : project bundle, published to the Project Deployer when available

Idempotent: re-running it only adds what is missing (use --build to (re)build the flow
part and run the evaluation, --bundle to export / publish the bundle and API package).

Usage (with the Python environment shipped with DSS, which includes dataikuapi):
    set DSS_API_KEY=<personal API key>
    python scripts/setup_mlops.py [--host http://localhost:11200] [--build] [--bundle]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dataikuapi

PROJECT_KEY = "CREDITOPS360"
ZONE = "5 Monitoring"
MODEL_NAME = "pd_model"
CONNECTION = "filesystem_managed"
RECENT_FILTER = "WEEK_NUM >= 80"  # last 12 production weeks (~165k cases)
ENRICH_ROWS = 2000
MES_NAME = "mes_pd_model"
API_SERVICE = "credit_scoring"
ENDPOINT = "predict_default"
ENDPOINT_BY_ID = "predict_default_by_id"
NON_FEATURES = ("case_id", "target", "WEEK_NUM", "MONTH")
BUNDLE_ID = "bundle_v1"
API_PACKAGE = "v1"


def api_key() -> str:
    key = os.environ.get("DSS_API_KEY")
    if not key:
        f = Path.home() / ".creditops_dss_key"
        key = f.read_text().strip() if f.exists() else ""
    if not key:
        sys.exit("No API key: set DSS_API_KEY or create ~/.creditops_dss_key")
    return key


def get_zone(project, name):
    for z in project.get_flow().list_zones():
        if z.name == name:
            return z
    return project.get_flow().create_zone(name)


def saved_model_id(project) -> str:
    for m in project.list_saved_models():
        if m["name"] == MODEL_NAME:
            return m["id"]
    sys.exit(f"saved model {MODEL_NAME} not found: run the training first")


def recipe_names(project) -> set:
    return {r["name"] for r in project.list_recipes()}


def ensure_sampling(project, zone, name, src, out, selection: dict) -> None:
    """Filter/Sample recipe src -> out (managed CSV dataset)."""
    if name not in recipe_names(project):
        builder = project.new_recipe("sampling", name)
        builder.with_input(src)
        builder.with_new_output(out, CONNECTION, format_option_id="CSV")
        recipe = builder.create()
        print(f"recipe {name}: created")
    else:
        recipe = project.get_recipe(name)
    settings = recipe.get_settings()
    settings.get_recipe_params()["selection"].update(selection)
    settings.save()
    zone.add_item(recipe)
    zone.add_item(project.get_dataset(out))


def ensure_mes(project, zone) -> str:
    for mes in project.list_model_evaluation_stores():
        if mes.get_settings().get_raw().get("name") == MES_NAME:
            mes_id = mes.mes_id
            break
    else:
        mes_id = project.create_model_evaluation_store(MES_NAME).mes_id
        print(f"model evaluation store {MES_NAME}: created ({mes_id})")
    zone.add_item(project.get_model_evaluation_store(mes_id))
    return mes_id


def ensure_metrics_dataset(project, name) -> None:
    if name not in {d["name"] for d in project.list_datasets()}:
        b = project.new_managed_dataset(name)
        b.with_store_into(CONNECTION, format_option_id="CSV")
        b.create()


def ensure_evaluate(project, zone, sm_id, mes_id) -> None:
    name = "evaluate_pd_model"
    metrics_ds = f"{MES_NAME}_metrics"
    ensure_metrics_dataset(project, metrics_ds)
    if name not in recipe_names(project):
        builder = project.new_recipe("evaluation", name)
        builder.with_input_model(sm_id)
        builder.with_input("hc_prod_recent")
        builder.with_output_metrics(metrics_ds)
        builder.with_output_evaluation_store(mes_id)
        recipe = builder.build()
        print(f"recipe {name}: created")
    else:
        recipe = project.get_recipe(name)
    settings = recipe.get_settings()
    payload = settings.obj_payload
    payload["outputProbabilities"] = True
    payload["dontComputePerformance"] = False
    payload["metrics"] = ["auc", "logLoss", "precision", "recall", "f1", "accuracy", "calibrationLoss"]
    # Evaluation labels help to compare evaluations in the MES
    payload["labels"] = [{"key": "scope", "value": "hc_prod weeks 80-91"},
                         {"key": "model", "value": MODEL_NAME}]
    settings.save()
    # Align the metrics dataset schema with the metrics computed by the recipe
    updates = recipe.compute_schema_updates()
    if updates.any_action_required():
        updates.apply()
    zone.add_item(recipe)
    zone.add_item(project.get_dataset(metrics_ds))


def build(project, mes_id, inputs: bool) -> None:
    targets = ["hc_prod_recent", "hc_enrich_cases"] if inputs else []
    for ds in targets + [f"{MES_NAME}_metrics"]:
        print(f"building {ds} ...", flush=True)
        project.get_dataset(ds).build(job_type="NON_RECURSIVE_FORCED_BUILD")
    report_mes(project, mes_id)


def report_mes(project, mes_id) -> None:
    mes = project.get_model_evaluation_store(mes_id)
    me = mes.get_latest_model_evaluation()
    if me is None:
        print("MES: no evaluation yet")
        return
    metrics = me.get_full_info().get_raw().get("metrics", {})
    keep = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print("last evaluation", me.full_id, json.dumps(keep))
    # Drift of the evaluated data against the training data of the model (reference)
    drift = me.compute_drift()
    if drift.drift_model_result is not None:
        acc = drift.drift_model_result.drift_model_accuracy
        print(f"data drift: drift model accuracy {acc.value:.4f} "
              f"[{acc.lower_confidence_interval:.4f}, {acc.upper_confidence_interval:.4f}] p-value {acc.pvalue:.4g}")
    if drift.prediction_drift_result is not None:
        pd_ = drift.prediction_drift_result
        print(f"prediction drift: PSI {pd_.psi} KS p-value {pd_.ks} chi2 p-value {pd_.chiSquare}")


def enrich_rows(project, n: int) -> tuple[list, list]:
    """First n rows of hc_enrich_cases (columns, rows)."""
    ds = project.get_dataset("hc_enrich_cases")
    cols = [c["name"] for c in ds.get_settings().get_raw()["schema"]["columns"]]
    rows = []
    for row in ds.iter_rows():
        rows.append(row)
        if len(rows) == n:
            break
    return cols, rows


def json_value(v):
    """Dates are sent as ISO 8601 strings, like an API client would."""
    return v.strftime("%Y-%m-%dT%H:%M:%S.000Z") if hasattr(v, "strftime") else v


def ensure_api_service(project, sm_id) -> None:
    if API_SERVICE not in {s["id"] for s in project.list_api_services()}:
        project.create_api_service(API_SERVICE)
        print(f"api service {API_SERVICE}: created")
    service = project.get_api_service(API_SERVICE)
    settings = service.get_settings()
    for ep_id in (ENDPOINT, ENDPOINT_BY_ID):
        if not any(e["id"] == ep_id for e in settings.get_raw()["endpoints"]):
            settings.add_prediction_endpoint(ep_id, sm_id)
    endpoints = {e["id"]: e for e in settings.get_raw()["endpoints"]}

    cols, rows = enrich_rows(project, 3)
    features = [c for c in cols if c not in NON_FEATURES]

    # 1) predict_default: the caller sends the full feature vector of the application
    ep = endpoints[ENDPOINT]
    ep["enrichMapping"] = []
    ep["testQueries"] = []
    for row in rows:
        rec = dict(zip(cols, row))
        feats = {f: json_value(rec[f]) for f in features if rec[f] not in (None, "")}
        ep["testQueries"].append({"name": f"case {rec['case_id']}", "q": {"features": feats}})

    # 2) predict_default_by_id: the caller only sends case_id; the features are added by
    # query enrichment from hc_enrich_cases (Bundled: copied into the API package and
    # loaded into a SQL connection of the API node).
    ep = endpoints[ENDPOINT_BY_ID]
    ep["enrichMapping"] = [{
        "datasetRef": "hc_enrich_cases",
        "packagingType": "BUNDLED_TOCONNECTION",
        "on": [{"resourceLookupCol": "case_id", "inputLookupKey": "case_id"}],
        "columnsMapping": {f: f for f in features},
        "missingLookupKeyBehavior": "FAIL",
        "notFoundBehavior": "FAIL",
        "multiMatchBehavior": "KEEP_FIRST",
    }]
    ep["returnPostEnrichData"] = False
    ep["testQueries"] = [{"name": f"case {dict(zip(cols, r))['case_id']}",
                          "q": {"features": {"case_id": int(dict(zip(cols, r))["case_id"])}}}
                         for r in rows]
    settings.save()


def play_test_queries(client, project, endpoint=ENDPOINT) -> None:
    """Runs the endpoint test queries on the Design node dev server (no API node needed).

    Uses an internal route of the API Designer: it requires a browser session, an API key
    is refused (401). Kept for reference; the UI button "Run test queries" does the same.
    """
    ep = next(e for e in project.get_api_service(API_SERVICE).get_settings().get_raw()["endpoints"]
              if e["id"] == endpoint)
    resp = client._session.post(
        f"{client.host}/dip/api/lambda-services/play-test-queries",
        data={"projectKey": PROJECT_KEY, "serviceId": API_SERVICE, "endpointId": endpoint,
              "testType": "predict", "queries": json.dumps(ep["testQueries"])})
    resp.raise_for_status()
    for tq, r in zip(ep["testQueries"], resp.json().get("responses", [])):
        print(tq["name"], json.dumps(r.get("response", r), default=str)[:400])


def ensure_bundle(client, project) -> None:
    # Default bundle content: flow, recipes, saved model active version, API service,
    # MES definition (no dataset data: the automation node rebuilds from its connections)
    existing = [b["id"] for b in project.list_exported_bundles().get("bundles", [])]
    try:
        if BUNDLE_ID not in existing:
            project.export_bundle(BUNDLE_ID, release_notes="CreditOps 360: pd_model, monitoring, MES")
            print(f"bundle {BUNDLE_ID}: exported")
        print("bundle publish:", project.publish_bundle(BUNDLE_ID))
    except Exception as exc:  # e.g. LicenseRestrictionException on a Design-only trial
        print("bundle export/publish failed:", exc)
    service = project.get_api_service(API_SERVICE)
    if API_PACKAGE not in [p["id"] for p in service.list_packages()]:
        service.create_package(API_PACKAGE, release_notes="predict_default + case_id enrichment")
        print(f"api package {API_PACKAGE}: created")
    try:
        print("api publish:", service.publish_package(API_PACKAGE))
    except Exception as exc:
        print("api publish failed:", exc)
    try:
        print("project deployer infras:", client.get_projectdeployer().list_infras(as_objects=False))
        print("api deployer infras:", client.get_apideployer().list_infras(as_objects=False))
    except Exception as exc:
        print("deployer infras:", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11200")
    parser.add_argument("--build", action="store_true", help="run the evaluation (Evaluate recipe)")
    parser.add_argument("--build-inputs", action="store_true", help="also rebuild hc_prod_recent / hc_enrich_cases")
    parser.add_argument("--test", action="store_true", help="play the API endpoint test queries")
    parser.add_argument("--bundle", action="store_true", help="export/publish bundle and API package")
    args = parser.parse_args()

    client = dataikuapi.DSSClient(args.host, api_key())
    project = client.get_project(PROJECT_KEY)
    zone = get_zone(project, ZONE)
    sm_id = saved_model_id(project)

    ensure_sampling(project, zone, "compute_hc_prod_recent", "hc_prod", "hc_prod_recent", {
        "samplingMethod": "FULL",
        "filter": {"enabled": True, "distinct": False, "expression": RECENT_FILTER,
                   "uiData": {"mode": "CUSTOM", "conditions": []}},
    })
    ensure_sampling(project, zone, "compute_hc_enrich_cases", "hc_prod_recent", "hc_enrich_cases", {
        "samplingMethod": "HEAD_SEQUENTIAL", "maxRecords": ENRICH_ROWS,
        "filter": {"enabled": False, "distinct": False},
    })
    mes_id = ensure_mes(project, zone)
    ensure_evaluate(project, zone, sm_id, mes_id)
    if args.build:
        build(project, mes_id, args.build_inputs)
    ensure_api_service(project, sm_id)
    if args.test:
        for ep_id in (ENDPOINT, ENDPOINT_BY_ID):
            try:
                play_test_queries(client, project, ep_id)
            except Exception as exc:
                print(f"test queries {ep_id}: not run ({exc}); use 'Run test queries' in the UI")
    if args.bundle:
        ensure_bundle(client, project)
    print("done")


if __name__ == "__main__":
    main()
