import dill
from src.exception import CustomException
from src.logger import logging
import sys
import os 
def model_saver(obj,path):
    try:
        with open(path,'wb') as f:
            dill.dump(obj,f)
    except Exception as e:
        logging.info("error in dumping model")
        raise(CustomException(e,sys))
