import logging
import os
from datetime import datetime

log_date = f"{datetime.now().strftime('%m_%d_%Y')}"

log_folder_date = os.path.join(os.getcwd(),"logs",log_date)

os.makedirs(log_folder_date,exist_ok=True)

log_file_path = os.path.join(log_folder_date,log_date+".log")

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s]",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info("divide by zero")