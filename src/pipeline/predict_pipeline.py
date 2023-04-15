import pandas as pd 
import os
from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass



@dataclass
class predict_pipeline_config:
    preprocessor_obj = os.path.join(os.getcwd(),'src','models','preprocessing.pkl')
    model_obj = os.path.join(os.getcwd(),'src','models','best_model.pkl')


class Predict_pineline:
    def __init__(self):
        self.predict_config = predict_pipeline_config()
    
    def predict(self,features):

        preprocessor=load_object(self.predict_config.preprocessor_obj)
        model = load_object(self.predict_config.model_obj)
        preprocessed_data=preprocessor.predict(features)
        pred_res = model.predict(preprocessed_data)

        return pred_res

    def custom_data(self):
        pass



if __name__ == '__main__':
    pass
