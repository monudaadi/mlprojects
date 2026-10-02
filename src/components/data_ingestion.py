# Import necessary libraries
import os
import sys
from src.exception import CustomException   # Custom exception handling class
from src.logger import logging              # Custom logger for tracking execution
import pandas as pd                         # For handling datasets
from sklearn.model_selection import train_test_split  # For splitting dataset
from dataclasses import dataclass           # For creating configuration class

from src.components.data_transformation import DataTransformation  # For data transformation
from src.components.data_transformation import DataTransformationConfig  # For data transformation configuration

# Configuration class to store file paths
@dataclass
class DataIngestionConfig:
    # Paths where train, test, and raw data will be stored
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

# Main class for data ingestion
class DataIngestion:
    def __init__(self):
        # Initialize configuration
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Step 1: Read dataset from source file
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Step 2: Create artifacts directory if it doesn’t exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Step 3: Save raw dataset into artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Step 4: Perform train-test split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Step 5: Save train and test datasets into artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Step 6: Return file paths for train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # If any error occurs, raise custom exception
            raise CustomException(e, sys)

if __name__ == "__main__":
    # If this script is run directly, initiate data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)