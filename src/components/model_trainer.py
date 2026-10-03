# Import required libraries
import os
import sys
from dataclasses import dataclass

# Machine learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Custom exception and logging utilities
from src.exception import CustomException
from src.logger import logging

# Utility functions for saving models and evaluating them
from src.utils import save_object, evaluate_models

# Configuration class to store model file path
@dataclass
class ModelTrainerConfig:
    # Path where the best trained model will be saved
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

# Main class for training models
class ModelTrainer:
    def __init__(self):
        # Initialize configuration
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data.")
            
            # Step 1: Separate features (X) and target (y) from train and test arrays
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All columns except last → features
                train_array[:, -1],   # Last column → target
                test_array[:, :-1],   # Features for test set
                test_array[:, -1]     # Target for test set
            )

            # Step 2: Define a dictionary of models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {  
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # Step 3: Evaluate all models using utility function
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params=params)

            # Step 4: Find the best model based on R2 score
            best_model_score = max(model_report.values())  # Highest R2 score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # Step 5: Check if the best model is good enough (R2 > 0.6)
            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score above 0.6", sys)
            
            logging.info(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score}")

            # Step 6: Save the best model to artifacts folder
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Handle any errors with custom exception
            raise CustomException(e, sys)
