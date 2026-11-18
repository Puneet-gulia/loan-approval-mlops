"""Smoke-test a running local/container/cloud service using fictional inputs."""
import argparse
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

SAMPLE = {'no_of_dependents': 2, 'education': 'Graduate', 'self_employed': 'No',
          'income_annum': 5000000, 'loan_amount': 15000000, 'loan_term': 10,
          'cibil_score': 780, 'residential_assets_value': 8000000,
          'commercial_assets_value': 3000000, 'luxury_assets_value': 10000000,
          'bank_asset_value': 4000000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    base = args.url.rstrip('/')
    for route in ['/', '/docs', '/health']:
        with urlopen(base + route, timeout=60) as response:
            assert response.status == 200
    def predict(data):
        request = Request(base + '/predict', data=json.dumps(data).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=60) as response:
            assert response.status == 200
            return json.load(response)
    result = predict(SAMPLE)
    assert result['prediction'] in ['Approved', 'Rejected']
    assert len(result['tips']) == 3
    weak = predict({**SAMPLE, 'cibil_score': 400})
    strong = predict({**SAMPLE, 'cibil_score': 800})
    assert strong['approval_probability'] > weak['approval_probability']
    try:
        predict({**SAMPLE, 'cibil_score': 901})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError('Invalid input was accepted')
    print(json.dumps({'base_url': base, 'status': 'passed', 'checks': [
        'UI', 'docs', 'health', 'valid prediction', 'domain logic', 'HTTP 422'],
        'sample_prediction': result}, indent=2))


if __name__ == '__main__':
    main()
