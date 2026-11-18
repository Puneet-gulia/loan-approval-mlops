"""Validated FastAPI serving with one startup load and durable JSONL logging."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent
EXAMPLE = {'no_of_dependents': 2, 'education': 'Graduate', 'self_employed': 'No',
           'income_annum': 5000000, 'loan_amount': 15000000, 'loan_term': 10,
           'cibil_score': 780, 'residential_assets_value': 8000000,
           'commercial_assets_value': 3000000, 'luxury_assets_value': 10000000,
           'bank_asset_value': 4000000}


class LoanRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', json_schema_extra={'example': EXAMPLE})
    no_of_dependents: int = Field(ge=0, le=5)
    education: Literal['Graduate', 'Not Graduate']
    self_employed: Literal['Yes', 'No']
    income_annum: int = Field(ge=100000, le=9900000, description='Annual income in whole INR')
    loan_amount: int = Field(ge=300000, le=39500000, description='Requested loan in whole INR')
    loan_term: int = Field(ge=2, le=20, multiple_of=2, description='Even number of years')
    cibil_score: int = Field(ge=300, le=900)
    residential_assets_value: int = Field(ge=0, le=29100000)
    commercial_assets_value: int = Field(ge=0, le=19400000)
    luxury_assets_value: int = Field(ge=0, le=39200000)
    bank_asset_value: int = Field(ge=0, le=14700000)


class Prediction(BaseModel):
    prediction: Literal['Approved', 'Rejected']
    approval_probability: float = Field(ge=0, le=1)
    tips: list[str]
    request_id: str
    model_version: str


def make_tips(data):
    tips = [
        ('Check your credit report for errors and keep repayments on time.' if data.cibil_score < 650
         else 'Keep repayments on time to maintain your credit history.'),
        ('Compare a smaller loan request with your annual income and existing obligations.'
         if data.loan_amount / data.income_annum > 3 else
         'Check the proposed repayment against your regular expenses and existing obligations.'),
        'Verify income and asset documents before submitting an application.'
    ]
    return tips


@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.environ.get('MODEL_PATH', ROOT / 'model/model.joblib'))
    if not model_path.exists():
        raise RuntimeError('Model missing. Run python train.py --compare first.')
    app.state.model = joblib.load(model_path)  # once per process, outside request handler
    app.state.model_version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    log_path = Path(os.environ.get('PREDICTION_LOG', ROOT / 'logs/predictions.jsonl'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # One worker per container. Rotated records also feed the drift check.
    app.state.log_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    yield
    app.state.log_handler.close()


app = FastAPI(title='Loan Approval Prediction', version='1.0.0', lifespan=lifespan,
              description='CSL236 educational model. Predicts dataset labels, not lending eligibility.')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Exclude raw inputs so nonstandard JSON NaN/Infinity cannot break the
    # validation response serializer and turn a client error into HTTP 500.
    return JSONResponse(status_code=422, content={'detail': [
        {'loc': list(error['loc']), 'msg': error['msg'], 'type': error['type']}
        for error in exc.errors()]})


@app.get('/health')
def health(request: Request):
    return {'status': 'ok', 'model_loaded': True, 'model_version': request.app.state.model_version}


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(ROOT / 'static/index.html')


@app.post('/predict', response_model=Prediction)
def predict(data: LoanRequest, request: Request):
    values = data.model_dump()
    model = request.app.state.model
    row = pd.DataFrame([values])
    probability = float(model.predict_proba(row)[0, list(model.classes_).index(1)])
    label = 'Approved' if probability >= .5 else 'Rejected'
    request_id = str(uuid4())
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'request_id': request_id,
              'model_version': request.app.state.model_version, 'inputs': values,
              'prediction': label, 'approval_probability': probability}
    # Return an explicit failure rather than silently serving unlogged predictions.
    try:
        handler = request.app.state.log_handler
        log_record = logging.LogRecord('predictions', logging.INFO, __file__, 0,
                                       json.dumps(record, allow_nan=False), (), None)
        # Write under the handler's lock. File errors must propagate (emit normally swallows them).
        with handler.lock:
            if handler.shouldRollover(log_record):
                handler.doRollover()
            if handler.stream is None:
                handler.stream = handler._open()
            handler.stream.write(log_record.getMessage() + '\n')
            handler.stream.flush()
    except OSError as exc:
        raise HTTPException(status_code=503, detail='Prediction logging unavailable. Please retry later.') from exc
    return Prediction(prediction=label, approval_probability=probability, tips=make_tips(data),
                      request_id=request_id, model_version=request.app.state.model_version)
