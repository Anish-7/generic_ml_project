from src.exception import CustomException
from src.logger import logging
from src.utils import model_saver

from dataclasses import dataclass
import os

from sklearn.linear_model  import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor , AdaBoostRegressor , GradientBoostingRegressor
from xgboost import XGBRegressor

from src.utils import eval_model
import sys

from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_transformation

@dataclass
class Trainer_config:
    model_save_path : str = os.path.join(os.getcwd(),"src","models")


class Model_trainer:
    def __init__(self) :
        logging.info("Initialising the config")
        self.trainer_config = Trainer_config()

    def initialse_trainer(self,train_set ,test_set):
        logging.info("Initialising the models dict")
        models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(),
            'random_forest':RandomForestRegressor(),
            'ada_boost': AdaBoostRegressor(),
            'gradientboost':GradientBoostingRegressor(),
            'xgboost':XGBRegressor()
        }
        logging.info("Initialising the params for hyperparameter tuning")
        params ={
            "decision_tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                    },
                "random_forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradientboost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear_regression":{
            
                },
                "xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "ada_boost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
        logging.info("spliting train and test data to respective features and target using unpacking")
        x_train,x_test,y_train,y_test = (train_set[:,:-1],test_set[:,:-1] , train_set[:,-1],test_set[:,-1])
        
        try:
            logging.info("evaluating models to find best models")
            report:dict= eval_model(x_train,x_test,y_train,y_test,models,params)
        except Exception as e:
            logging.info("error in evalution of model")
            raise CustomException(e,sys)

        logging.info("getting best model")
        try:
            logging.info("sorting to get best model")
            best_score = max(sorted(report.values()))
            
            if best_score <0.6:
                logging.info("best model is less than 0.6 which is not good")
                print("best model is less than 0.6 which is not good")
    
            best_model_name = list(report.keys())[
                    list(report.values()).index(best_score)
                ]
            best_model = models[best_model_name]
            #doubt here we ge best model... but  do we get best parms also as we got in grid search cv
            logging.info("saving best model")
            os.makedirs(self.trainer_config.model_save_path,exist_ok=True)
            model_saver(obj=best_model,path=os.path.join(self.trainer_config.model_save_path,'best_model.pkl'))         
            logging.info("best model saved")
        
        except Exception as e :
            logging.info("error in getting best model")
            raise CustomException(e,sys)

        return report
        ## next time try get fitted gs model with best params from utils

if __name__ == '__main__':
    train_path, test_path =Data_Ingestion().initiate_ingestion()
    dt =Data_transformation()
    train,test,pre_path = dt.initialse_transform(train_path, test_path)
    # print(train.shape)
    mt = Model_trainer()
    report = mt.initialse_trainer(train,test)
     
    print(report)