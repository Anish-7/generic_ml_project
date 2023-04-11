import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class Data_ingestion_config:
    raw_data_path : str = os.path.join(os.getcwd(),"artifacts","raw_data")
    train_data_path : str= os.path.join(os.getcwd(),"artifacts","train_data")
    test_data_path  :str = os.path.join(os.getcwd(),"artifacts","test_data")

class Data_Ingestion:
    def __init__(self):
        self.data_ingestion_config = Data_ingestion_config()

    def initiate_ingestion(self):
        logging.info('Initiated data ingestion')

        try:
            logging.info("Reading the data as dataframe")
            data = pd.read_csv(os.path.join(os.getcwd(),"notebooks","data","stud.csv"))
            logging.info("Finished reading the data as dataframe")
        
        except Exception as e:
            logging.info("Error occured during reading the file")
            raise CustomException(e,sys)

        try:
            logging.info("Splitting data into train test split")
            logging.info("making directories for train and test split ")
            os.makedirs(self.data_ingestion_config.train_data_path,exist_ok=True)
            os.makedirs(self.data_ingestion_config.test_data_path,exist_ok=True)            
            train_set,test_set = train_test_split(data,test_size=0.3)
            logging.info("saving train and test split ")
            train_set.to_csv(os.path.join(self.data_ingestion_config.train_data_path,"train.csv"),header=True,index=False)
            test_set.to_csv(os.path.join(self.data_ingestion_config.test_data_path,"test.csv"),header=True,index=False)

        except Exception as e:
            logging.info("error occured while splitting train and test")
            raise CustomException(e,sys)

        return (os.path.join(self.data_ingestion_config.train_data_path,"train.csv"),os.path.join(self.data_ingestion_config.test_data_path,"test.csv"))

if __name__ == "__main__":
    obj = Data_Ingestion()
    obj.initiate_ingestion()