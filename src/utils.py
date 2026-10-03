import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object will be saved.
        obj: The Python object to be saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        # Create the directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to the specified file path
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)   

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    """
    Evaluate multiple machine learning models and return their R2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target values.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target values.
        models (dict): A dictionary of model names and their corresponding model instances.
        params (dict, optional): Hyperparameter grids keyed by model name.

    Returns:
        dict: A dictionary containing model names and their corresponding R2 scores.
        """
    try:
        model_report = {}

        for model_name, model in models.items():
            model_params = (params or {}).get(model_name, {})
            if model_params:
                gs = GridSearchCV(model, model_params, cv=3)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                models[model_name] = model

            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2_square_test = r2_score(y_test, y_test_pred)
            model_report[model_name] = r2_square_test

        return model_report

    except Exception as e:
        raise CustomException(e, sys)