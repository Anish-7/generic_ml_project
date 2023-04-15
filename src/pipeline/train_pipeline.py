from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_transformation
from src.components.model_trainer import Model_trainer


class Train_pipeline:

    def __init__(self):
        pass

    def initiate_trainer(self):

        logging.info('initiating data ingestion')
        train_path,test_path =Data_Ingestion().initiate_ingestion()
        logging.info('initiating data ingestion')
        train,test,preprocessor_obj_path =Data_transformation().initialse_transform(train_path,test_path)
        logging.info('training models')
        report = Model_trainer().initialse_trainer(train_set=train,test_set=test)

        return report
    

if __name__ =='__main__':

    report = Train_pipeline().initiate_trainer()
    print(report)
