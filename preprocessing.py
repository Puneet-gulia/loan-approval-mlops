"""Shared, serializable preprocessing. Every learned value is fitted per fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

NUMERIC = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
           'cibil_score', 'residential_assets_value', 'commercial_assets_value',
           'luxury_assets_value', 'bank_asset_value']
FEATURES = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
ENGINEERED = ['cibil_band', 'term_band', 'income_band', 'loan_to_income', 'total_assets']
ASSETS = ['residential_assets_value', 'commercial_assets_value',
          'luxury_assets_value', 'bank_asset_value']


def clean_frame(frame):
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    for column in frame.select_dtypes(include=['object', 'string']).columns:
        frame[column] = frame[column].str.strip()
    return frame


class LoanFeatures(TransformerMixin, BaseEstimator):
    """Keep 11 inputs and derive 5 features, following the mid-term notebook.

    Invalid negative residential assets are a cleaning assumption, not a
    confirmed missing-value code. Serving rejects negatives; training imputes
    these 28 historical records using training-only medians.
    """

    def fit(self, X, y=None):
        data = clean_frame(X)[FEATURES]
        numeric = data[NUMERIC].astype(float)
        numeric['residential_assets_value'] = numeric['residential_assets_value'].mask(
            numeric['residential_assets_value'] < 0)
        self.medians_ = numeric.median()
        self.income_edges_ = np.quantile(numeric['income_annum'].dropna(), [.25, .5, .75])
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, 'medians_')
        data = clean_frame(X)[FEATURES].copy()
        for col in NUMERIC:
            data[col] = data[col].astype(float)
        col = 'residential_assets_value'
        data[col] = data[col].mask(data[col] < 0)
        data[NUMERIC] = data[NUMERIC].fillna(self.medians_)
        data['education'] = data['education'].map({'Not Graduate': 0, 'Graduate': 1})
        data['self_employed'] = data['self_employed'].map({'No': 0, 'Yes': 1})
        # These are project bins, not official lending eligibility thresholds.
        data['cibil_band'] = np.searchsorted([550, 650, 750], data.cibil_score, side='right')
        data['term_band'] = np.searchsorted([5, 10], data.loan_term, side='left')
        data['income_band'] = np.searchsorted(self.income_edges_, data.income_annum, side='left')
        data['loan_to_income'] = data.loan_amount / data.income_annum
        data['total_assets'] = data[ASSETS].sum(axis=1)
        for col in ['residential_assets_value', 'commercial_assets_value']:
            data[col] = np.sqrt(data[col])
        if not np.isfinite(data.to_numpy(dtype=float)).all():
            raise ValueError('Preprocessing produced invalid values; check the input data.')
        return data[FEATURES + ENGINEERED].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + ENGINEERED, dtype=object)
