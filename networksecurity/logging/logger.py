import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path = os.path.join( os.path.dirname(os.path.abspath(__file__) ), 'logs' , LOG_FILE)   


                
os.makedirs(log_path , exist_ok=True)
log_file_path = os.path.join(log_path,LOG_FILE)


logging.basicConfig(
    filename=log_file_path , # or use log_file_path variable
    level=logging.INFO,  # or DEBUG, ERROR, etc.
    format='[%(asctime)s]: %(lineno)d : %(name)s : %(levelname)s: %(message)s',
   
)

