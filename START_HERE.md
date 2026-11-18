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

The docs/ folder contains the nine-page report and eleven-slide editable PowerPoint. Both are drafts for final submission because team details, public links and hosted evidence are missing. Complete submission.json with the real details and update the documents. Rename the report and slides as required by the guidelines.

## Remaining work

Docker is not installed on the machine used to build this package. Its saved GitHub login is invalid. The files for Docker, GitHub Actions and deployment are included, but image size, a green hosted run and a public URL cannot be verified until those tools/accounts are available. The README gives the commands and handoff checklist.

Do not submit the ZIP as a claim that cloud deployment has already completed. Publish the repository, verify Docker and CI, deploy, and add authentic screenshots and public links first.
