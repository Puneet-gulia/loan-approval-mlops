# Viva rehearsal

## Five-minute demo

- Open the deployed UI, load an example, and request an estimate.
- Open /docs and submit cibil_score=901 to demonstrate HTTP 422.
- Open MLflow and compare the four runs.
- Show the latest real green GitHub Actions run after publishing.

## Questions to prepare

**Why Random Forest?** It tied for the highest cross-validation Rejected-class F1. The 100-tree depth-6 model matched the larger forest, so the smaller configuration was selected. Explain that choice with docs/experiments.csv.

**Why emphasize Rejected F1?** Approved labels are the majority. Accuracy alone can hide poor performance on the Rejected class. F1 considers precision and recall, but it is not a substitute for a real lending cost function.

**Where is preprocessing fitted?** Inside each training fold through the scikit-learn Pipeline. Medians, quartiles and scaler statistics never fit the validation fold or holdout. The final pipeline fits all training rows after model selection.

**Why retain 11 inputs?** The end-term asks for at least five. The mid-term reduced features more aggressively. This version keeps the original predictors and adds derived columns for 16 transformed features.

**What happens to NaN?** Historical negative residential assets become missing and use the fitted median. The public API requires every input and rejects null, missing, NaN and Infinity with 422.

**Why two requirements files?** Training needs MLflow and testing tools. The serving image only needs the API and inference dependencies, which reduces its footprint.

**Does 99.53% accuracy prove it will work in a bank?** No. This dataset has sharp decision boundaries, its provenance is limited, and the holdout was examined during the mid-term. Independent data, calibration and subgroup checks are needed for real use.

**Why serialize the pipeline?** The API must apply exactly the same transforms as training. Loading a classifier without the fitted transforms would produce inconsistent predictions.

**Where is the model loaded?** Once per process in the FastAPI lifespan startup function. Requests reuse app.state.model.

**How is drift detected?** Logged raw numeric inputs are compared with training means/std after 30 records. The flags are heuristic and require review. They do not prove that predictive accuracy has fallen.

**What are the tips?** Deterministic rules, separate from the model. They are not causal explanations of why an application received a label.

**How do CI and deployment differ?** CI installs, trains, tests and builds on GitHub. Deployment runs the serving container on a host. The supplied files implement both configurations, but the team must execute and verify them with its accounts.

**What did each member do?** Answer with actual work and commits. Names and contributions have deliberately not been invented.
