import pytest
import mlflow.sklearn
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

@pytest.fixture
def production_model():
    """Fixture to load the model from MLflow before tests run."""
    model_uri = "models:/rf3_model@production"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def test_model_loads_successfully(production_model):
    """Test 1: Ensures the model is registered and loads without crashing."""
    assert production_model is not None, "Model failed to load from MLflow registry."

def test_model_prediction_output(production_model):
    """Test 2: Ensures the model outputs a 0 or 1 when given a dummy row."""
    
    dummy_data = pd.DataFrame({
        "feature_1": [25],
        "feature_2": [50000.0],
        "feature_3": [1]
    })
    
    prediction = production_model.predict(dummy_data)
    
    assert len(prediction) == 1, "Model should return exactly one prediction."
    assert prediction[0] in [0, 1], f"Prediction should be 0 or 1, but got {prediction[0]}"