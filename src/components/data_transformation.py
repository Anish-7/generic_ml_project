import os 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import Data_Ingestion
from dataclasses import dataclass
import sys
import pandas as pd
import numpy as np
from src.utils import model_saver

@dataclass
class Data_transform_config:

    data_transform_pkl : str= os.path.join(os.getcwd(),"src","models")


class Data_transformation:

    def __init__(self):
        self.transform_model_path = Data_transform_config()


    def get_transformer_obj(self):
        categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
        numeric_columns = ['reading_score','writing_score']

        try:
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="mean")),
                ("normalisation",StandardScaler())  
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder())
                ]
            )
        
            preprocessor  = ColumnTransformer(
                transformers=[
                            ("numeric_transform",num_pipeline,numeric_columns),
                            ("cat_transform",cat_pipeline,categorical_columns)
                            ]
            )
            return preprocessor

        
        except Exception as e :
            logging.info("error in creating transoformer object")
            raise CustomException(e,sys)
    
    def initialse_transform(self,train_set_path,test_set_path):

        try:
            logging.info("loading preprocessor object")
            preprocessor_obj = self.get_transformer_obj()
            logging.info("loading preprocessor object")        
            
            logging.info("loading test and train data")
            train_data = pd.read_csv(train_set_path)
            test_data = pd.read_csv(test_set_path)
            logging.info("loaded preprocessor object")
            
            target = ['math_score']
            
            logging.info("splitting to attributes and target variable")
            input_train_features = train_data.drop(target,axis=1)        
            input_test_features = test_data.drop(target,axis=1)

            target_train_feature = train_data[target]
            target_test_feature = test_data[target]

            logging.info("using preprocessing object fit_transform train set")
            train_features_arr =preprocessor_obj.fit_transform(input_train_features)
            logging.info("using preprocessing object transform test set")
            test_features_arr = preprocessor_obj.transform(input_test_features)

            logging.info("concating the transformed train/test attributes and target into array")
            train = np.c_[train_features_arr,np.array(target_train_feature)]
            test = np.c_[test_features_arr,np.array(target_test_feature)]

            logging.info("saving preprocessing model")
            os.makedirs(self.transform_model_path.data_transform_pkl,exist_ok=True)
            model_saver(obj=preprocessor_obj,path=os.path.join(self.transform_model_path.data_transform_pkl,"preprocessing.pkl"))

            return train ,test , self.transform_model_path.data_transform_pkl

        except Exception as e :
            logging.info("error in preprocessing")
            raise CustomException(e,sys)
    
if __name__ == '__main__':
    train_path, test_path =Data_Ingestion().initiate_ingestion()
    dt =Data_transformation()
    train,test,pre_path = dt.initialse_transform(train_path, test_path)

    print()