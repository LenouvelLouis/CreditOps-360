# Scenario weekly_retrain, Python step: retrains the PD model only when the
# monitoring raised an alert (score drift or Gini drop) on the production weeks.
# Manual runs can pass {"dry_run": true} to log the decision without training.
import dataiku
from dataiku.scenario import Scenario

MODEL_NAME = "pd_model"

scenario = Scenario()
dry_run = str((scenario.get_trigger_params() or {}).get("dry_run", "false")).lower() == "true"

summary = dataiku.Dataset("hc_stability_summary").get_dataframe().iloc[0]
psi_alerts = int(summary["weeks_psi_alert"])
gini_alerts = int(summary["weeks_gini_alert"])
print(f"monitoring: {psi_alerts} week(s) with PSI alert, {gini_alerts} week(s) with Gini alert, "
      f"stability score {summary['stability_score']:.4f}")

if psi_alerts == 0 and gini_alerts == 0:
    decision = "skipped"
    print("decision: no retrain, model stable")
elif dry_run:
    decision = "retrain_required_dry_run"
    print("decision: retrain required (dry run, model not trained)")
else:
    decision = "retrained"
    print(f"decision: retrain, training saved model {MODEL_NAME}")
    project = dataiku.api_client().get_default_project()
    model_id = next(m["id"] for m in project.list_saved_models() if m["name"] == MODEL_NAME)
    # hc_train is already up to date (built by the previous steps)
    scenario.train_model(model_id, build_mode="NON_RECURSIVE_FORCED_BUILD")

scenario.set_scenario_variables(retrain_decision=decision)
