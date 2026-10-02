## feature engineering and data transformation

import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # Path to save the preprocessor object
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        # Initialize the configuration for data transformation
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for creating and returning a preprocessor object.
        It defines the preprocessing steps for both numerical and categorical features.
        """
        try:
            # Define numerical and categorical columns
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            # Create a pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
                    ('scaler', StandardScaler())                     # Scale features to have mean=0 and variance=1
                ]
            )

            # Create a pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent value
                    ('one_hot_encoder', OneHotEncoder()),                  # Convert categorical variables to one-hot encoding
                    ('scaler', StandardScaler(with_mean=False))            # Scale features (without centering)
                ]
            )

            logging.info(f"Numerical columns standard scaling completed./n Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns one-hot encoding completed./n Categorical columns: {categorical_columns}")

            # Combine both pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns
                    ('cat_pipeline', cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical columns
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation process.
        It reads the training and testing datasets, applies the preprocessing steps,
        and saves the preprocessor object for future use.
        """
        try:
            # Read the training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object.")
            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()

            # Separate input features and target variable for training data
            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Fit and transform the training data, transform the testing data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            # Combine transformed features with target variable for both training and testing datasets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object to a file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
    
            logging.info("Preprocessing object saved successfully.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)