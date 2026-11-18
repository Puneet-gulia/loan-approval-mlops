import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, EXAMPLE, LoanRequest
from preprocessing import LoanFeatures, FEATURES


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('PREDICTION_LOG', str(tmp_path / 'predictions.jsonl'))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['model_loaded'] is True


def test_ui_loads(client):
    response = client.get('/')
    assert response.status_code == 200
    assert 'Loan approval studio' in response.text
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/style.css').status_code == 200


def test_valid_prediction_and_logging(client):
    result = client.post('/predict', json=EXAMPLE)
    assert result.status_code == 200
    payload = result.json()
    assert payload['prediction'] in ['Approved', 'Rejected']
    assert 0 <= payload['approval_probability'] <= 1
    assert len(payload['tips']) == 3
    path = Path(app.state.log_handler.baseFilename)
    record = json.loads(path.read_text().splitlines()[-1])
    assert record['inputs'] == EXAMPLE
    assert record['request_id'] == payload['request_id']
    assert record['model_version'] == payload['model_version']


def test_domain_stronger_credit_profile_scores_higher(client):
    weak = client.post('/predict', json={**EXAMPLE, 'cibil_score': 400}).json()
    strong = client.post('/predict', json={**EXAMPLE, 'cibil_score': 800}).json()
    assert strong['approval_probability'] > weak['approval_probability']


@pytest.mark.parametrize('field,value', [
    ('no_of_dependents', -1), ('no_of_dependents', 6),
    ('education', 'Doctorate'), ('self_employed', 'Maybe'),
    ('income_annum', 99999), ('income_annum', 9900001),
    ('loan_amount', 299999), ('loan_amount', 39500001),
    ('loan_term', 0), ('loan_term', 22), ('loan_term', 3),
    ('cibil_score', 299), ('cibil_score', 901),
    ('residential_assets_value', -1), ('residential_assets_value', 29100001),
    ('commercial_assets_value', -1), ('commercial_assets_value', 19400001),
    ('luxury_assets_value', -1), ('luxury_assets_value', 39200001),
    ('bank_asset_value', -1), ('bank_asset_value', 14700001),
    ('cibil_score', '780'), ('cibil_score', True), ('loan_term', 10.0),
    ('income_annum', None), ('education', 1)
])
def test_contract_invalid_values_return_422(client, field, value):
    assert client.post('/predict', json={**EXAMPLE, field: value}).status_code == 422


@pytest.mark.parametrize('field', FEATURES)
def test_missing_field_is_422(client, field):
    payload = EXAMPLE.copy()
    del payload[field]
    assert client.post('/predict', json=payload).status_code == 422


def test_extra_field_is_422(client):
    assert client.post('/predict', json={**EXAMPLE, 'loan_status': 'Approved'}).status_code == 422


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_raw_json_is_422(client, token):
    body = json.dumps(EXAMPLE).replace('780', token)
    response = client.post('/predict', content=body, headers={'Content-Type': 'application/json'})
    assert response.status_code == 422


def test_malformed_json_is_422(client):
    assert client.post('/predict', content='{oops', headers={'Content-Type': 'application/json'}).status_code == 422


@pytest.mark.parametrize('high', [False, True])
def test_exact_contract_boundaries_are_valid(client, high):
    payload = EXAMPLE.copy()
    for name, field in LoanRequest.model_json_schema()['properties'].items():
        if field.get('type') == 'integer':
            payload[name] = field['maximum' if high else 'minimum']
    assert client.post('/predict', json=payload).status_code == 200


def test_openapi_documents_contract(client):
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json').json()
    assert len(schema['components']['schemas']['LoanRequest']['required']) == 11


def test_model_loads_once_at_startup(tmp_path, monkeypatch):
    monkeypatch.setenv('PREDICTION_LOG', str(tmp_path / 'load-once.jsonl'))
    original_load = joblib.load
    with patch('app.joblib.load', wraps=original_load) as load:
        with TestClient(app) as client:
            assert load.call_count == 1
            for _ in range(2):
                assert client.post('/predict', json=EXAMPLE).status_code == 200
            assert load.call_count == 1


def test_logging_failure_returns_503(client):
    with patch.object(app.state.log_handler, 'shouldRollover', side_effect=OSError('disk full')):
        assert client.post('/predict', json=EXAMPLE).status_code == 503


def test_preprocessing_is_training_only_and_roundtrips(tmp_path):
    train = pd.DataFrame([{**EXAMPLE, 'income_annum': 1000000, 'residential_assets_value': 1000000},
                          {**EXAMPLE, 'income_annum': 2000000, 'residential_assets_value': -100000}])
    transform = LoanFeatures().fit(train)
    before = transform.medians_.copy()
    incoming = pd.DataFrame([{**EXAMPLE, 'income_annum': 9900000, 'residential_assets_value': -100000}])
    output = transform.transform(incoming)
    pd.testing.assert_series_equal(transform.medians_, before)
    assert output.residential_assets_value.iloc[0] == 1000
    assert output.shape == (1, 16)
    file = tmp_path / 'features.joblib'
    joblib.dump(transform, file)
    np.testing.assert_allclose(joblib.load(file).transform(incoming), output)
