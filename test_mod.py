import pytest
import os
from pathlib import Path
import sys
sys.path.append([str(Path(os.getcwd()).parent),str(Path(os.getcwd()))])
from flask_app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_prediction(client):
    input_data ={
        'age':40,
        'estimated_salary':52222
    }


    response = client.post('/predict',json=input_data)
    assert response.status_code == 200
    response_data = json.loads(response.data)

    assert 'prediction' in response_data
    assert response_data['prediction'] in ['Not likely to purchase','Likely to purchase']