import logging
import os.path
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%H%M',time.localtime(time.time()))
log_path = os.path.join(os.getcwd() ,'Logs')
log_name = os.path.join(log_path,rq+'.log')
logfile = log_name
file_handler = logging.FileHandler(logfile,'w')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)

# add the logger to handler

logger.addHandler(file_handler)
