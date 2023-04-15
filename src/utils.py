import dill
from src.exception import CustomException
from src.logger import logging
import sys
import os 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def model_saver(obj,path):
    try:
        with open(path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        logging.info("error in dumping model")
        raise(CustomException(e,sys))

def eval_model(x_train,x_test,y_train,y_test,models,params):

    reports= {}
    for model_name in models:
        logging.info(f"evaluating {model_name}")
        model = models[model_name]
        params_model = params[model_name]
        logging.info(f"finding best hyper parameter using grid search cv {model_name}")
        gs = GridSearchCV(model,params_model,)
        gs.fit(x_train,y_train)
        logging.info(f"setting best params {model_name}")
        model.set_params(**gs.best_params_)
        model.fit(x_train,y_train)
        
        ypred = model.predict(x_test)
        logging.info(f"reporting matrics of {model_name}")
        reports[model_name] = r2_score(y_pred=ypred,y_true=y_test)

    return reports

def load_object(path):
    logging.info(f"loading objects")
    try:
        with open(path,'rb') as f:
            obj = dill.load(f)
            return obj
    except Exception as e :
        logging.info('error loading object')
        raise CustomException(e,sys)

if __name__ == '__main__':
    pass