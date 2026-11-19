# Start here

This is the local CSL236 end-term implementation. The original files in Downloads were left unchanged.

1. Open README.md for setup and measured results.
2. Create the Python 3.11 environment and install requirements.txt.
3. Run `uvicorn app:app --port 8000`, then visit http://127.0.0.1:8000.
4. Use "Load example", then "Estimate loan outcome".
5. Run `python -m pytest -v --tb=short` to reproduce the checks.
6. Run `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` to see the four recorded experiments.

The trained model is included, so you can start the app before retraining. Run `python train.py --compare` to reproduce the comparison.

## Documents

The docs/ folder contains the 10-page report and 12-slide editable PowerPoint. Both include the public links and hosted evidence. All three members and section AI-B are included. Complete actual contribution allocations before submission. Individually named submission copies are in docs/submission/.

## Remaining work

GitHub is published at https://github.com/Puneet-gulia/loan-approval-mlops. Docker and CI verification passed; image sizes and evidence are in README.md. The free cloud deployment is live at https://loan-approval-mlops-w8p2.onrender.com. Its public UI, API docs, health, valid predictions and HTTP 422 passed verification. Only actual contribution allocations remain to be completed.

Complete each member's contribution allocation before submission. Recheck the live URL before demo day. The free service may take 50 seconds or more to wake, and its prediction log files reset when the instance is replaced.
