import sys
import logging
from networksecurity.logging import logger
def get_error_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in file: {file_name}, at line: {line_number}, error message: {str(error)}"

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # super().__init__(error_message)
        self.error_message = error_message
        _,_,exc_tb = error_detail.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
     return "error occurred in script name [{0}] line no [{1}] error message [{2}]".format(
    self.file_name, self.lineno, str(self.error_message)
)


if __name__ =='__main__':
   
   try:
      logger.logging.info('ENTERED THE TRIED BLOCK')

      a = 1/0
      print('this will not be printed' , a)


   except Exception as e:
      raise NetworkSecurityException(e,sys)    




