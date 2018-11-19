#!/bin/python
import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from waitress import serve
from ua_parser import user_agent_parser

app = Flask(__name__)

@app.route('/status')
def status():
    return "alive"

@app.route('/brand-fraud', methods=['POST'])
def brand_fraud():
    data = pd.DataFrame.from_dict(request.get_json(silent=True))
    brand_fraud = joblib.load('model/brand_fraud.pkl')

    prediction = brand_fraud.predict_proba(data)
    return jsonify(prediction[:, 1].tolist())

def create_app():
    return serve(app)