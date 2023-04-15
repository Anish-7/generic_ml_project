import pandas as pd 
import os
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

@dataclass
class predict_pipeline_config:
    preprocessor_obj = os.path.join(os.getcwd(),'src','models','preprocessing.pkl')
    model_obj = os.path.join(os.getcwd(),'src','models','best_model.pkl')


class Predict_pipeline:
    def __init__(self):
        self.predict_config = predict_pipeline_config()
    
    def predict(self,sample_data):
        try :
            logging.info('loading model and preprocessor objects')
            preprocessor=load_object(self.predict_config.preprocessor_obj)
            model = load_object(self.predict_config.model_obj)

            logging.info('transforming given data')        
            preprocessed_data=preprocessor.transform(sample_data)
            logging.info('predicting given data')  
            pred_res = model.predict(preprocessed_data)
        
        except Exception as e:
            logging.info('error in predicting')

            raise CustomException(e,sys)

        return pred_res

class Custom_data:
    def __init__(self,gender,ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender =gender
        self.ethnicity = ethnicity
        self.parental_level_of_education =parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

        
    def get_features(self):
        # keys should be same as data transformation pipeline
        features_dict:dict = {"gender":self.gender,
                                  "race_ethnicity":self.ethnicity,   
                                  "parental_level_of_education":self.parental_level_of_education,
                                  "lunch":self.lunch,
                                  "test_preparation_course":self.test_preparation_course,
                                  "reading_score":self.reading_score, 
                                  "writing_score":self.writing_score
                                  }

        features = pd.DataFrame(features_dict,index=[0]) # index[0]is given because it has single sample of scalar values
 
        return features


if __name__ == '__main__':
    pass
