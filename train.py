"""Reproducible training, fold-safe validation and SQLite-backed MLflow runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import LoanFeatures, FEATURES, NUMERIC, clean_frame

ROOT = Path(__file__).resolve().parent
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['random_forest', 'logistic_regression'], default='random_forest')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--compare', action='store_true', help='Run four configurations across two algorithms')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/loan_approval_dataset.csv')
    args = parser.parse_args()
    if args.n_estimators < 1 or args.max_depth < 1 or args.C <= 0:
        parser.error('n_estimators, max_depth and C must be positive')
    return args


def main():
    args = parse_args()
    data = clean_frame(pd.read_csv(args.data))
    if len(data) < 1000 or not set(FEATURES + ['loan_status']).issubset(data.columns):
        raise ValueError('Dataset must contain >=1000 rows and all required columns')
    y = data.loan_status.map({'Rejected': 0, 'Approved': 1})
    if y.isna().any():
        raise ValueError('Unexpected or missing target label')
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], y, test_size=.2, stratify=y, random_state=SEED)
    configs = ([{'algorithm': 'logistic_regression', 'C': .1},
                {'algorithm': 'logistic_regression', 'C': 1.0},
                {'algorithm': 'random_forest', 'n_estimators': 100, 'max_depth': 6},
                {'algorithm': 'random_forest', 'n_estimators': 200, 'max_depth': 10}]
               if args.compare else [dict(algorithm=args.algorithm, **(
                   {'C': args.C} if args.algorithm == 'logistic_regression' else
                   {'n_estimators': args.n_estimators, 'max_depth': args.max_depth}))])
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///' + str(ROOT / 'mlflow.db'))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('loan-approval')
    scoring = {'accuracy': 'accuracy', 'f1_rejected': make_scorer(f1_score, pos_label=0)}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows, candidates = [], []
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    for number, config in enumerate(configs, 1):
        estimator = (LogisticRegression(C=config['C'], max_iter=3000, random_state=SEED)
                     if config['algorithm'] == 'logistic_regression' else
                     RandomForestClassifier(n_estimators=config['n_estimators'],
                                            max_depth=config['max_depth'], random_state=SEED, n_jobs=1))
        pipeline = Pipeline([('features', LoanFeatures()), ('scale', StandardScaler()), ('model', estimator)])
        with mlflow.start_run(run_name=f"{config['algorithm']}-{number}") as run:
            mlflow.log_params({**config, 'random_state': SEED, 'cv_folds': 3,
                               'raw_features': len(FEATURES), 'test_fraction': .2})
            mlflow.set_tags({'dataset_sha256': digest, 'selection_metric': 'cv_f1_rejected',
                             'preprocessing': 'midterm-derived; fit inside each CV fold'})
            scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score='raise')
            metrics = {'cv_accuracy': float(scores['test_accuracy'].mean()),
                       'cv_f1_rejected': float(scores['test_f1_rejected'].mean()),
                       'cv_f1_std': float(scores['test_f1_rejected'].std())}
            mlflow.log_metrics(metrics)
            row = {'run_id': run.info.run_id, **config, **metrics}
            rows.append(row)
            candidates.append(pipeline)
            print(json.dumps(row), flush=True)
    # The held-out set never participates in model selection.
    winner = max(range(len(rows)), key=lambda i: (rows[i]['cv_f1_rejected'], rows[i]['cv_accuracy']))
    best = clone(candidates[winner]).fit(X_train, y_train)
    predictions = best.predict(X_test)
    test_metrics = {'test_accuracy': accuracy_score(y_test, predictions),
                    'test_f1_rejected': f1_score(y_test, predictions, pos_label=0),
                    'test_f1_approved': f1_score(y_test, predictions, pos_label=1)}
    with mlflow.start_run(run_id=rows[winner]['run_id']):
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag('selected', 'true')
    out = ROOT / 'model'
    out.mkdir(exist_ok=True)
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    joblib.dump(best, out / 'model.joblib')
    model_sha = hashlib.sha256((out / 'model.joblib').read_bytes()).hexdigest()
    metadata = {'trained_at_utc': datetime.now(timezone.utc).isoformat(),
                'selected_run': rows[winner], 'test_metrics': test_metrics,
                'dataset_sha256': digest, 'model_sha256': model_sha,
                'rows': len(data), 'train_rows': len(X_train), 'test_rows': len(X_test),
                'input_features': FEATURES, 'transformed_features': best['features'].get_feature_names_out().tolist(),
                'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
                'class_order': ['Rejected', 'Approved'],
                'majority_baseline_accuracy': float((y_test == y_train.mode().iloc[0]).mean())}
    if hasattr(best['model'], 'feature_importances_'):
        metadata['feature_importance'] = dict(zip(metadata['transformed_features'], best['model'].feature_importances_.tolist()))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(docs / 'experiments.csv', index=False)
    reference = X_train[NUMERIC].astype(float).copy()
    reference['residential_assets_value'] = reference.residential_assets_value.mask(reference.residential_assets_value < 0)
    reference = reference.fillna(best['features'].medians_)
    (out / 'reference.json').write_text(json.dumps({c: {'mean': float(reference[c].mean()),
               'std': float(reference[c].std(ddof=0))} for c in NUMERIC}, indent=2) + '\n')
    print('Selected using CV:', json.dumps(rows[winner]))
    print('Held-out evaluation:', json.dumps(test_metrics))


if __name__ == '__main__':
    main()
