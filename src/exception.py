import sys 
from src.logger import logging
def error_message_detail(error,error_detail:sys):

    _,_,exc_tb = error_detail.exc_info()
    error_message = f"error occured in python script : {exc_tb.tb_frame.f_code.co_filename} line number : {exc_tb.tb_lineno} error_message {error}"
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(error)
        self.error_message= error_message_detail(error=error,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    try:
        a= 4/0
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)