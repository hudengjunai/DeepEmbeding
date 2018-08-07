import requests
import numpy as np

host = 'http://hpc5.yud.io:8080/ping'
ret = requests.post(host)
print(ret)